"""Deterministic six-stage execution with durable smoke/batch and human gates."""
from __future__ import annotations

import json
import shutil
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from .. import ensemble, preparation
from ..common import error_details, fingerprint, read_composite, save_json, turnIntoPureText
from .budget import Budget
from .contracts import ResearchSpecification, TaskKey, create_run, file_hash, validate_manifest, verify_inputs
from .lease import RunLease
from .report import write_report
from .runtime import ControlStop, RunContext
from .store import Store, StoreError


def begin(request: dict, config, runs_dir="runs", *, dry_run=False, run_id=None) -> Path:
    spec = ResearchSpecification.from_config(request, config)
    run = create_run(spec, runs_dir, run_id)
    with Store(run, create=True) as store:
        store.initialize(spec.fingerprint)
        store.set_state("PREFLIGHT", "explicit specification frozen; no API request yet")
        verify_inputs(run)
        _plan_tasks(store, spec.data, run.name)
        store.set_state("INPUT_READY", "isolated input snapshots verified")
        store.export_views()
    runtime = RunContext(run, config)
    try:
        write_report(runtime, "plan_report")
    finally:
        runtime.close()
    if not dry_run:
        execute(run, config)
    return run


def _plan_tasks(store, spec, run_id):
    for paper in spec["papers"]:
        for model in spec["models"]:
            for question in spec["questions"]:
                for round_index in spec["rounds"]:
                    key = TaskKey(run_id, "examiner", paper["paper_uid"], question["index"],
                                  source_model=model["model"], round_index=round_index)
                    store.put_task(key.task_id, key.stage, key.to_dict())


def _prepare(runtime):
    root = runtime.run_dir
    for paper in runtime.spec["papers"]:
        key = runtime.key("prepare", paper_uid=paper["paper_uid"])
        target = root / "outputs" / "prepared" / (paper["paper_uid"] + ".md")
        metadata = target.with_suffix(".meta.json")
        with runtime.task_scope(key):
            if runtime.valid(key):
                if not turnIntoPureText(target).strip():
                    raise ValueError("prepared article body is empty")
                runtime.completed(key, [target, metadata], cached=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = root / "inputs" / (paper["paper_uid"] + paper["suffix"])
            converter = None
            if paper["suffix"] == ".pdf":
                if not runtime.spec["allow_pdf_resources"]:
                    runtime._gate("pdf_resources", key.task_id, "PDF model resource preparation was not explicitly allowed")
                converter = version("marker-pdf")
                pdf_dir = root / "checkpoints" / "pdf_inputs" / paper["paper_uid"]
                pdf_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, pdf_dir / source.name)
                # A damaged derived Markdown must be rebuilt, not accepted by the legacy file-exists check.
                target.unlink(missing_ok=True)
                preparation.convert_pdfs_to_markdown(pdf_dir, target.parent)
            else:
                shutil.copyfile(source, target)
            if not turnIntoPureText(target).strip():
                raise ValueError("prepared article body is empty")
            save_json(metadata, dict(paper_uid=paper["paper_uid"], input_sha256=paper["sha256"],
                                     markdown_sha256=file_hash(target), converter=converter))
            runtime.completed(key, [target, metadata])


def _scope_config(runtime, paper_uids, round_index):
    spec, root = runtime.spec, runtime.run_dir
    scope_hash = fingerprint(sorted(paper_uids))
    markdown_dir = root / "checkpoints" / "scopes" / scope_hash
    markdown_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {uid + ".md" for uid in paper_uids}
    if {p.name for p in markdown_dir.glob("*.md")} - expected_names:
        raise ValueError("unexpected paper in the frozen execution scope")
    for uid in paper_uids:
        if not runtime.valid(runtime.key("prepare", paper_uid=uid)):
            raise ValueError("prepared input changed or lacks a valid preparation receipt")
        source = root / "outputs" / "prepared" / (uid + ".md")
        shutil.copyfile(source, markdown_dir / source.name)
    cfg = dict(markdown_dir=str(markdown_dir), pdf_dir=str(root / "checkpoints" / "no_pdf_conversion"),
               domain_knowledge=spec["domain_knowledge"], composite_prefix="LUMINA",
               questions=[q["index"] for q in spec["questions"]])
    for field, folder in dict(examiner_output="examiner", composite_dir="composite", embedding_dir="embeddings",
                              crosser_dir="cross", ensemble_dir="ensemble").items():
        cfg[field] = str(root / "outputs" / f"R{round_index:02d}" / folder)
    original = runtime.config
    config = SimpleNamespace(DOMAINS={spec["domain"]: cfg}, FULL_LLM_POOL=original.FULL_LLM_POOL,
                             SELECTED_KEYS=original.SELECTED_KEYS, LLM_SETTINGS=original.LLM_SETTINGS,
                             EMBEDDING_MODEL=original.EMBEDDING_MODEL, RUN=spec["run"] | {"round_index": round_index})
    return config, scope_hash


def _execute_scope(runtime, paper_uids):
    import run_pipeline

    fields = dict(examiner="examiner_output", composite="composite_dir", embeddings="embedding_dir",
                  cross="crosser_dir", ensemble="ensemble_dir")
    for round_index in runtime.spec["rounds"]:
        config, scope_hash = _scope_config(runtime, paper_uids, round_index)
        cfg = config.DOMAINS[runtime.spec["domain"]]
        for stage, field in fields.items():
            key = runtime.key(stage, round_index=round_index, chunk=scope_hash)
            with runtime.task_scope(key):
                # Core semantic checks still run; paid task/response caches prevent extra calls.
                run_pipeline._run(runtime.spec["domain"], stage, config, runtime=runtime)
                files = sorted(p for p in Path(cfg[field]).rglob("*") if p.is_file() and p.suffix in {".csv", ".xlsx", ".npy", ".json"})
                checkpoint = runtime.run_dir / "checkpoints" / "stages" / key.task_id[:16]
                checkpoint.mkdir(parents=True, exist_ok=True)
                receipt = checkpoint / "stage_result.json"
                if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")).get("task_id") != key.task_id:
                    raise ValueError("stage checkpoint directory identity collision")
                artifacts = []
                for source in files:
                    destination = checkpoint / source.relative_to(cfg[field])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    artifacts.append(destination)
                # Legitimate stages with no candidates still leave a verifiable result receipt.
                save_json(receipt, dict(task_id=key.task_id, stage=stage, scope=sorted(paper_uids), round_index=round_index,
                                        artifacts=[str(p.relative_to(checkpoint)) for p in artifacts]))
                runtime.completed(key, artifacts + [receipt])


def _final_qc(runtime):
    import run_pipeline

    papers = [p["paper_uid"] for p in runtime.spec["papers"]]
    extraction = [t for t in runtime.store.tasks("examiner") if t["payload"].get("paper_uid")]
    expected = len(papers) * len(runtime.spec["models"]) * len(runtime.spec["questions"]) * len(runtime.spec["rounds"])
    if len(extraction) != expected or any(not runtime.store.valid_task(t["task_id"]) for t in extraction):
        raise ValueError("incomplete or changed extraction artifacts")
    outputs = []
    for round_index in runtime.spec["rounds"]:
        config, scope_hash = _scope_config(runtime, papers, round_index)
        cfg = config.DOMAINS[runtime.spec["domain"]]
        models = run_pipeline.build_llm_dicts(config)
        names = [m["model"].split("/")[-1] for m in models.values()]
        run_pipeline.validate_composites(runtime.spec["domain"], cfg, models, config.LLM_SETTINGS, config.RUN)
        composite_key = runtime.key("composite", round_index=round_index, chunk=scope_hash)
        ensemble_key = runtime.key("ensemble", round_index=round_index, chunk=scope_hash)
        if not runtime.valid(composite_key) or not runtime.valid(ensemble_key):
            raise ValueError("composite or ensemble stage proof missing or changed")
        baseline_root = runtime.run_dir / "checkpoints" / "stages" / composite_key.task_id[:16]
        for question in runtime.spec["questions"]:
            filename = f"LUMINA_Q{question['index']:02d}.xlsx"
            frame = read_composite(Path(cfg["composite_dir"]) / filename)
            baseline = read_composite(baseline_root / filename)
            original = frame.loc[:, baseline.columns]
            if not original.equals(baseline):
                raise ValueError("composite scientific content differs from its baseline receipt")
            if config.RUN["min_cross_scores"]:
                ensemble.validate_cross_coverage(frame, names)
        files = sorted(Path(cfg["ensemble_dir"]).rglob("*.xlsx"))
        expected_files = len(runtime.spec["questions"]) * 2 * (1 + len(config.RUN["min_cross_scores"]))
        if len(files) != expected_files or any(file.stat().st_size == 0 for file in files):
            raise ValueError("ensemble outputs incomplete")
        ensemble_root = runtime.run_dir / "checkpoints" / "stages" / ensemble_key.task_id[:16]
        if any(file_hash(file) != file_hash(ensemble_root / file.relative_to(cfg["ensemble_dir"])) for file in files):
            raise ValueError("ensemble output differs from its stage receipt")
        outputs.extend(dict(path=str(file.relative_to(runtime.run_dir)), sha256=file_hash(file)) for file in files)
    cross_tasks = [task for task in runtime.store.tasks("cross") if task["payload"].get("paper_uid")]
    if any(not runtime.store.valid_task(task["task_id"]) for task in cross_tasks):
        raise ValueError("cross vote artifact differs from its recorded receipt")
    outputs = [dict(path=str(file.relative_to(runtime.run_dir)), sha256=file_hash(file))
               for file in sorted((runtime.run_dir / "outputs").rglob("*")) if file.is_file()]
    return dict(passed=True, extraction_expected=expected, outputs=outputs,
                scientific_validation="NOT_EVALUATED")


def execute(run_dir, config) -> dict:
    result = _execute_once(run_dir, config, allow_recovery=True)
    if result.pop("recovery_queued", False):
        result = _execute_once(run_dir, config, allow_recovery=False)
    return result


def _diagnose_failure(runtime, error, *, allow_recovery):
    from .advisor import apply_proposal, diagnose
    from .contracts import ActionProposal

    failed = [task for task in runtime.store.tasks() if task["status"] in {"failed", "unresolved"} and
              task["payload"].get("paper_uid") and task["stage"] in {"examiner", "cross", "embeddings", "evidence_embedding"}]
    target = max(failed, key=lambda task: task["updated_at"])["task_id"] if failed else runtime.run_dir.name
    fallback = ActionProposal("request_human", target, "Inspect the saved error and receipt; automatic recovery did not resolve it")
    if not allow_recovery:
        return apply_proposal(runtime, fallback)
    try:
        return apply_proposal(runtime, diagnose(runtime, error, target))
    except ControlStop:
        return {"queued": False}  # Diagnosis is subject to the same durable budget/unknown gates.
    except (ValueError, RuntimeError) as exc:
        runtime.store.event("advisor_error", {"error": runtime.redact(str(exc)), "target": target})
        return apply_proposal(runtime, fallback)


def _execute_once(run_dir, config, *, allow_recovery) -> dict:
    run = Path(run_dir).resolve()
    with RunLease(run) as lease:
        try:
            runtime = RunContext(run, config)
        except Exception as exc:
            detail = error_details(exc, config.LLM_SETTINGS)
            try:
                with Store(run) as failed_store:
                    if failed_store.snapshot()["run"]["state"] not in {"DONE", "FATAL_ERROR"}:
                        failed_store.set_state("FATAL_ERROR", detail)
                    failed_store.event("error_initialization", {"error": detail})
                    failed_store.export_views()
            except StoreError:
                pass  # An unusable authority stays intact; never create a replacement ledger.
            result = dict(state="FATAL_ERROR", run_id=run.name, error=detail, scientific_validation="NOT_EVALUATED")
            save_json(run / "reports" / "fatal_report.json", result)
            return result
        runtime.heartbeat = lease.touch
        store = runtime.store
        qc = None
        try:
            state = store.snapshot()["run"]["state"]
            if state == "DONE":
                qc = json.loads((run / "checkpoints" / "final_qc.json").read_text(encoding="utf-8"))
                for artifact in qc["outputs"]:
                    file = (run / artifact["path"]).resolve()
                    if not file.is_relative_to(run / "outputs") or not file.is_file() or file_hash(file) != artifact["sha256"]:
                        raise ValueError("archived production artifact missing or changed")
                _final_qc(runtime)
                return dict(state="DONE", run_id=run.name)
            if state == "FATAL_ERROR":
                return dict(state=state, run_id=run.name)
            recovered = runtime.budget.recover_unfinished()
            # Finish a durable human decision if the process stopped between approval and accounting.
            for gate in store.gates():
                if gate["kind"] == "unknown_request" and gate["status"] == "approved":
                    attempt = runtime.budget.attempt(gate["target"])
                    if attempt and attempt["status"] == "unknown" and not attempt["receipt"].get("reconciled"):
                        decision = gate["decision"]
                        runtime.budget.reconcile(gate["target"], decision["outcome"], decision["reason"])
            for attempt_id in recovered["marked_unknown"]:
                if not any(g["kind"] == "unknown_request" and g["target"] == attempt_id and g["status"] == "pending" for g in store.gates()):
                    store.create_gate("unknown_request", attempt_id, "worker stopped after dispatch; provider outcome unknown")
            if recovered["marked_unknown"] and state != "HUMAN_GATE":
                store.set_state("HUMAN_GATE", "unresolved requests require human reconciliation")
                state = "HUMAN_GATE"
            pending = [g for g in store.gates() if g["status"] == "pending"]
            if pending:
                return dict(state=state, run_id=run.name, pending_gates=[g["gate_id"] for g in pending])
            if state in {"PAUSED", "RECOVERABLE_ERROR", "HUMAN_GATE"}:
                store.clear_pause()
                state = store.set_state(store.snapshot()["run"]["resume_state"], "explicit resume after checks")["state"]
            if state == "INIT":
                state = store.set_state("PREFLIGHT", "resume interrupted initialization without resetting accounting")["state"]
            if state == "PREFLIGHT":
                verify_inputs(run)
                _plan_tasks(store, runtime.spec, run.name)
                state = store.set_state("INPUT_READY", "preflight recovered from frozen inputs")["state"]
            if state == "INPUT_READY":
                _prepare(runtime)
                state = store.set_state("SMOKE", "inputs prepared; executing smoke")['state']
            if state == "SMOKE":
                smoke = runtime.spec["smoke_paper_uids"]
                _execute_scope(runtime, smoke)
                write_report(runtime, "smoke_report", paper_uids=smoke)
                target = file_hash(run / "reports" / "smoke_report.json")
                store.create_gate("smoke", target, "review smoke_report.json and explicitly approve batch")
                state = store.set_state("HUMAN_GATE_SMOKE", "smoke finished; batch not approved")["state"]
                return dict(state=state, run_id=run.name)
            if state == "HUMAN_GATE_SMOKE":
                if not any(g["kind"] == "smoke" and g["status"] == "approved" for g in store.gates()):
                    raise ControlStop("batch requires explicit smoke approval")
                state = store.set_state("BATCH", "human-approved smoke report; execute remaining tasks")["state"]
            if state == "BATCH":
                _prepare(runtime)
                _execute_scope(runtime, [p["paper_uid"] for p in runtime.spec["papers"]])
                state = store.set_state("FINAL_QC", "all requested paper scopes executed")["state"]
            if state == "FINAL_QC":
                qc = _final_qc(runtime)
                save_json(run / "checkpoints" / "final_qc.json", qc)
                state = store.set_state("ARCHIVE", "final engineering quality checks passed")["state"]
            if state == "ARCHIVE":
                qc = json.loads((run / "checkpoints" / "final_qc.json").read_text(encoding="utf-8"))
                if not qc.get("passed"):
                    raise ValueError("final QC proof missing")
                # Revalidate canonical outputs after interrupted archiving.
                qc = _final_qc(runtime)
                state = store.set_state("DONE", "machine workflow complete; scientific validation remains independent")["state"]
            return dict(state=state, run_id=run.name)
        except ControlStop as exc:
            store.event("control_stop", {"reason": runtime.redact(str(exc))})
            if runtime.spec["advisor"] and allow_recovery and store.snapshot()["run"]["state"] == "HUMAN_GATE":
                _diagnose_failure(runtime, exc, allow_recovery=True)
            return dict(state=store.snapshot()["run"]["state"], run_id=run.name, reason=str(exc))
        except KeyboardInterrupt:
            store.request_pause()
            state = store.snapshot()["run"]["state"]
            if state != "PAUSED":
                store.set_state("PAUSED", "Ctrl-C: no promise to cancel an already dispatched request")
            return dict(state="PAUSED", run_id=run.name)
        except Exception as exc:
            detail = runtime.redact(f"{type(exc).__name__}: {exc}")
            store.event("error", {"error": detail})
            state = store.snapshot()["run"]["state"]
            if state not in {"FATAL_ERROR", "DONE"}:
                desired = "FATAL_ERROR" if isinstance(exc, StoreError) else "RECOVERABLE_ERROR"
                if state != desired:
                    store.set_state(desired, detail)
            qc = dict(passed=False, status="FAILED", error=detail)
            recovery = {"queued": False}
            if store.snapshot()["run"]["state"] == "RECOVERABLE_ERROR":
                recovery = _diagnose_failure(runtime, exc, allow_recovery=allow_recovery)
            return dict(state="FATAL_ERROR" if state == "DONE" else store.snapshot()["run"]["state"],
                        run_id=run.name, error=detail, recovery_queued=recovery["queued"])
        finally:
            try:
                try:
                    write_report(runtime, final_qc=qc)
                except StoreError as exc:
                    save_json(run / "reports" / "fatal_report.json",
                              dict(state="FATAL_ERROR", run_id=run.name, error=runtime.redact(str(exc)),
                                   scientific_validation="NOT_EVALUATED"))
            finally:
                runtime.close()


def approve(run_dir, gate_id: str, reason: str, *, config=None, outcome=None):
    run = Path(run_dir).resolve()
    with RunLease(run), Store(run) as store:
        manifest = validate_manifest(run, store.snapshot()["run"]["spec_hash"])
        gate = next((g for g in store.gates() if g["gate_id"] == gate_id), None)
        if gate is None or gate["status"] != "pending":
            raise ValueError("gate does not exist or has already been decided")
        decision = dict(reason=reason)
        if gate["kind"] == "smoke":
            if store.snapshot()["run"]["state"] != "HUMAN_GATE_SMOKE" or file_hash(run / "reports" / "smoke_report.json") != gate["target"]:
                raise ValueError("smoke report changed or run is not at the smoke gate")
            decision["scope"] = "batch"
            store.approve_gate(gate_id, decision)
        elif gate["kind"] == "unknown_request":
            if outcome not in {"not_executed", "executed_unknown"}:
                raise ValueError("unknown request approval requires an explicit provider outcome")
            decision["outcome"] = outcome
            budget = Budget(store, manifest["specification"])
            attempt = budget.attempt(gate["target"])
            if not attempt or attempt["status"] != "unknown":
                raise ValueError("gate does not identify an unresolved request")
            # Human decision is durable before acting on it; interruption cannot manufacture approval.
            store.approve_gate(gate_id, decision)
            budget.reconcile(gate["target"], outcome, reason)
        elif gate["kind"] == "confirmed_rejection":
            budget = Budget(store, manifest["specification"])
            attempt = budget.attempt(gate["target"])
            if not attempt or attempt["status"] != "rejected" or not attempt["receipt"].get("confirmed_not_executed"):
                raise ValueError("gate has no confirmed rejection receipt")
            store.approve_gate(gate_id, decision | {"scope": "retry unchanged request within original limits"})
        elif gate["kind"] == "diagnosis":
            if store.snapshot()["run"]["state"] not in {"HUMAN_GATE", "PAUSED"}:
                raise ValueError("diagnosis gate is not active")
            # This records resolution of an operational cause; other gates and frozen conditions remain enforced.
            store.approve_gate(gate_id, decision | {"scope": "resume unchanged specification after cause was resolved"})
        else:
            raise ValueError(f"{gate['kind']} cannot be bypassed by approval; resolve the cause or create a new frozen run")
        store.export_views()
        return dict(gate_id=gate_id, status="approved", decision=decision)
