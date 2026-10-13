"""Read-only live progress views for the supervised configuration workflow.

The SQLite ledger remains authoritative.  This module only opens it in read-only
mode, derives display counters from task receipts and events, and formats a
short human status line.  A malformed presentation artifact is a display gap,
never a worker failure.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import quote

from .agent.budget import _totals


def language_for(config=None, language=None) -> str:
    """Resolve the UI language without changing a run's frozen specification."""
    if language in {"zh", "en"}:
        return language
    project = getattr(config, "PROJECT", None)
    if isinstance(project, dict):
        return "zh" if project.get("language", "zh") == "zh" else "en"
    return "en"


def _json(value, default=None):
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return default
    return result


def _receipt_complete(task: dict) -> bool:
    """Use the ledger receipt only; report-time hashing remains the authority check."""
    if task.get("status") != "succeeded":
        return False
    artifacts = _json(task.get("artifacts"), [])
    if not isinstance(artifacts, list) or not artifacts:
        return False
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item.get("sha256"):
            return False
    return True


def _phase(state: str | None) -> str | None:
    if state in {"SMOKE", "HUMAN_GATE_SMOKE"}:
        return "trial"
    if state in {"BATCH", "FINAL_QC", "ARCHIVE", "DONE"}:
        return "batch"
    return None


def _scope_from(spec: dict, state: str | None) -> list[str]:
    papers = spec.get("papers", []) if isinstance(spec, dict) else []
    all_papers = [p.get("paper_uid") for p in papers if isinstance(p, dict) and p.get("paper_uid")]
    if state in {"SMOKE", "HUMAN_GATE_SMOKE"}:
        smoke = spec.get("smoke_paper_uids", [])
        return [uid for uid in smoke if uid in all_papers]
    return all_papers


def _event_payload(row) -> dict:
    payload = _json(row["payload"], {})
    return payload if isinstance(payload, dict) else {"malformed": True}


def _read_only_connection(path: Path, timeout: float) -> sqlite3.Connection:
    # quote() keeps drive letters and spaces valid in a Windows SQLite URI.
    uri = "file:" + quote(str(path.resolve()).replace("\\", "/"), safe="/:_") + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=max(0.01, float(timeout)))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute(f"PRAGMA busy_timeout={max(1, int(float(timeout) * 1000))}")
    return connection


def _read_manifest(run: Path) -> dict:
    try:
        value = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _accepted_rows(run: Path, spec: dict, round_index: int, frozen_artifacts=None) -> dict:
    """Read dual-threshold result files from frozen round artifacts only."""
    raw_scores = spec.get("run", {}).get("min_cross_scores", []) if isinstance(spec, dict) else []
    scores = [score for score in (raw_scores if isinstance(raw_scores, (list, tuple)) else [])
              if type(score) is int and score >= 0]
    domain = spec.get("domain", "")
    consensus = spec.get("run", {}).get("min_consensus_models") if isinstance(spec, dict) else None
    questions = [q.get("index") for q in spec.get("questions", [])
                 if isinstance(q, dict) and type(q.get("index")) is int and q["index"] > 0]
    result = {}
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - dependency is part of the normal profile
        return {f"MiniCross{score:02d}": {"rows": None, "previews": [], "error": str(exc)} for score in scores}
    artifact_paths = [Path(path) for path in (frozen_artifacts or []) if isinstance(path, str)]
    by_name = {path.name: run / path for path in artifact_paths}
    for score in scores:
        label = f"MiniCross{score:02d}"
        rows = 0
        previews = []
        errors = []
        found = False
        for question in questions:
            name = f"{domain}_Ensemble_Result_Q{question:02d}_{label}.xlsx"
            path = by_name.get(name)
            if path is None:
                continue
            found = True
            try:
                frame = pd.read_excel(path)
                rows += len(frame)
                for item in frame.head(max(0, 5 - len(previews))).to_dict("records"):
                    clean = {}
                    for key, value in item.items():
                        if isinstance(value, float) and math.isnan(value):
                            value = None
                        if hasattr(value, "item"):
                            value = value.item()
                        clean[str(key)] = value
                    previews.append(clean)
            except Exception as exc:  # presentation data is outside task authority
                errors.append(f"{path.relative_to(run).as_posix()}: {type(exc).__name__}: {exc}")
        entry = {"threshold": score, "verification_threshold": score,
                 "min_consensus_models": consensus,
                 "rows": None if errors or not found else rows, "previews": previews}
        if errors:
            entry["error"] = "; ".join(errors)
        result[label] = entry
    return result


def _active_wait(events: list[dict], attempts: list[dict]) -> dict | None:
    starts = {}
    finished = set()
    finished_kinds = set()
    for event in events:
        payload = event["payload"]
        wait_id = payload.get("wait_id") or payload.get("id") or payload.get("attempt_id")
        if event["kind"] == "wait_started":
            starts[wait_id or event["event_id"]] = (payload, event["at"])
        elif event["kind"] == "wait_finished":
            finished.add(wait_id)
            if wait_id is None:
                finished_kinds.add(payload.get("kind"))
    for wait_id, (payload, started_at) in reversed(list(starts.items())):
        if wait_id not in finished and payload.get("kind") not in finished_kinds:
            kind = payload.get("kind", "wait")
            if "rate" in str(kind).lower() or "sleep" in str(kind).lower():
                kind = "rate"
            return {"kind": kind, "model": payload.get("model"),
                    "elapsed_seconds": max(0.0, time.time() - started_at)}
    model_prepare = {}
    for event in events:
        if event["kind"] == "pdf_model_prepare_started":
            model_prepare[event["payload"].get("paper_uid")] = (event["payload"], event["at"])
        elif event["kind"] == "pdf_model_prepare_finished":
            model_prepare.pop(event["payload"].get("paper_uid"), None)
    if model_prepare:
        started_at = min(started for _, started in model_prepare.values())
        payloads = [payload for payload, _ in model_prepare.values()]
        return {"kind": "model_loading", "count": len(model_prepare),
                "model": next((payload.get("model") or payload.get("backend")
                                for payload in payloads), None),
                "elapsed_seconds": max(0.0, time.time() - started_at)}
    dispatched = [row for row in attempts if row.get("status") == "dispatched"]
    if dispatched:
        records = []
        for row in dispatched:
            receipt = _json(row.get("receipt"), {})
            records.append((receipt.get("model"), receipt.get("dispatched_at")))
        elapsed = [time.time() - started for _, started in records
                   if isinstance(started, (int, float)) and not isinstance(started, bool)]
        models = [model for model, _ in records if model]
        return {"kind": "model_response", "count": len(dispatched),
                "model": models[0] if len(models) == 1 else ",".join(models) if models else None,
                "elapsed_seconds": max(0.0, max(elapsed, default=0.0))}
    return None


def _snapshot(run: Path, connection: sqlite3.Connection, events: list[dict], accepted_cache=None) -> dict:
    manifest = _read_manifest(run)
    spec = manifest.get("specification", {}) if isinstance(manifest, dict) else {}
    run_row = connection.execute("SELECT * FROM run WHERE singleton=1").fetchone()
    if run_row is None:
        return {"available": False, "error": "run row unavailable", "events": events}
    tasks = [dict(row) for row in connection.execute("SELECT * FROM tasks ORDER BY task_id")]
    attempts = [dict(row) for row in connection.execute("SELECT * FROM attempts ORDER BY attempt_id")]
    accounting = _totals(connection)
    accounting.setdefault("known_cost_subtotal", accounting.get("known_cost"))
    state = run_row["state"]
    stage_starts = [e for e in events if e["kind"] == "stage_started"]
    stage_finishes = [e for e in events if e["kind"] == "stage_finished"]
    latest_finished = stage_finishes[-1]["payload"] if stage_finishes else {}
    current = None
    # Stages execute sequentially. A prior completion must not mask a later
    # invocation with the same task ID (trial/batch or interrupted resume).
    for event in reversed(events):
        if event["kind"] == "stage_started":
            current = event["payload"] | {"event_id": event["event_id"], "started_at": event["at"]}
            break
        if event["kind"] == "stage_finished":
            break
    phase = (current or {}).get("phase") or latest_finished.get("phase") or _phase(state)
    scope = list((current or {}).get("scope") or latest_finished.get("scope") or _scope_from(spec, state))
    round_index = (current or {}).get("round_index")
    if round_index is None:
        round_index = latest_finished.get("round_index")
    if round_index is None:
        rounds = spec.get("rounds", []) if isinstance(spec, dict) else []
        round_index = rounds[0] if rounds else None
    current_round = round_index
    models = spec.get("models", []) if isinstance(spec, dict) else []
    questions = spec.get("questions", []) if isinstance(spec, dict) else []
    def selected(task, stage):
        payload = _json(task.get("payload"), {})
        return (task.get("stage") == stage and isinstance(payload, dict)
                and payload.get("paper_uid") in scope and payload.get("round_index") == current_round)
    extraction_tasks = [t for t in tasks if selected(t, "examiner") and _json(t.get("payload"), {}).get("question")]
    extraction = dict(expected=len(scope) * len(models) * len(questions),
                      completed=sum(_receipt_complete(t) for t in extraction_tasks),
                      failed=sum(t.get("status") == "failed" for t in extraction_tasks),
                      unresolved=sum(t.get("status") == "unresolved" for t in extraction_tasks))
    cross_tasks = [t for t in tasks if selected(t, "cross") and _json(t.get("payload"), {}).get("candidate")]
    candidate_keys = set()
    pair_keys = set()
    completed_pairs = set()
    for task in cross_tasks:
        payload = _json(task.get("payload"), {})
        candidate_key = (payload.get("paper_uid"), payload.get("question"), payload.get("candidate"),
                         payload.get("source_model"))
        pair_key = candidate_key + (payload.get("verifier_model"),)
        candidate_keys.add(candidate_key)
        pair_keys.add(pair_key)
        if _receipt_complete(task):
            completed_pairs.add(pair_key)
    cross = dict(known=bool(cross_tasks), candidates=len(candidate_keys) if cross_tasks else None,
                 verifier_pairs=len(pair_keys) if cross_tasks else None,
                 completed_pairs=len(completed_pairs) if cross_tasks else None)
    now = time.time()
    stage_elapsed = None
    if current and isinstance(current.get("started_at"), (int, float)):
        stage_elapsed = max(0.0, now - current["started_at"])
    phase_started = None
    phase_states = {"trial": {"INPUT_READY", "SMOKE", "HUMAN_GATE_SMOKE"}, "batch": {"BATCH"}}
    for event in reversed(events):
        if event["kind"] == "state" and event["payload"].get("to") in phase_states.get(phase, set()):
            phase_started = event["at"]
            break
    context_stage = (current or {}).get("stage")
    if context_stage is None and stage_finishes:
        context_stage = stage_finishes[-1]["payload"].get("stage")
    context_round = current_round
    context_scope = set(scope)
    task_by_id = {task["task_id"]: task for task in tasks}
    context_start = next((event["event_id"] for event in reversed(stage_starts)
                          if event["payload"].get("stage") == context_stage
                          and event["payload"].get("phase") == phase
                          and event["payload"].get("round_index") == context_round
                          and set(event["payload"].get("scope", [])) == context_scope), 0)

    def cache_event_in_context(event):
        if event["kind"] not in {"cache_hit", "response_reused"} or event["event_id"] < context_start:
            return False
        task = task_by_id.get(event["payload"].get("task_id"))
        if task is None or task.get("stage") != context_stage:
            return False
        payload = _json(task.get("payload"), {})
        if not isinstance(payload, dict) or payload.get("paper_uid") not in context_scope:
            return False
        return payload.get("round_index") == context_round

    cache_reuse = sum(cache_event_in_context(event) for event in events)
    completed_rounds = []
    for event in events:
        if event["kind"] != "round_finished":
            continue
        payload = event["payload"]
        summary = dict(payload)
        summary["event_id"] = event["event_id"]
        try:
            if isinstance(payload.get("accepted"), dict):
                summary["accepted"] = payload["accepted"]
            else:
                frozen = payload.get("frozen_artifacts", {}).get("ensemble", [])
                cache_key = (int(payload["round_index"]), tuple(payload.get("scope", [])), tuple(frozen))
                if accepted_cache is not None and cache_key not in accepted_cache:
                    accepted_cache[cache_key] = _accepted_rows(run, spec, cache_key[0], frozen)
                summary["accepted"] = (accepted_cache or {}).get(cache_key) \
                    if accepted_cache is not None else _accepted_rows(run, spec, cache_key[0], frozen)
        except (TypeError, ValueError, OSError):
            summary["accepted"] = {}
        completed_rounds.append(summary)
    return {
        "available": True,
        "run_id": run_row["spec_hash"] and run.name,
        "state": state,
        "phase": phase,
        "current": current,
        "round": current_round,
        "scope": scope,
        "stage_elapsed_seconds": stage_elapsed,
        "phase_elapsed_seconds": None if phase_started is None else max(0.0, now - phase_started),
        "extraction": extraction,
        "cross": cross,
        "tasks": {"total": len(tasks), "completed": sum(t.get("status") == "succeeded" for t in tasks)},
        "cache_reuse": cache_reuse,
        "waiting": _active_wait(events, attempts),
        "cost": {field: accounting.get(field) for field in
                  ("total_cost", "known_cost", "known_cost_subtotal", "held_cost",
                   "unpriced_requests", "cost_complete")},
        "rounds": completed_rounds,
        "paths": [path for e in stage_finishes for path in e["payload"].get("paths", [])
                  if isinstance(path, str)],
        "events": events,
    }


class ProgressMonitor:
    """Incremental read-only monitor; failures are returned as unavailable data."""

    def __init__(self, run_dir, *, language="en", timeout=0.25, interval=1.0, stream=None):
        self.run_dir = Path(run_dir).resolve()
        self.language = language_for(language=language)
        self.timeout = timeout
        self.interval = interval
        self.stream = stream if stream is not None else sys.stderr
        self.last_event_id = 0
        self._events = []
        self._accepted_cache = {}
        self._last_line = None
        self._last_output = 0.0

    def poll(self) -> dict:
        connection = None
        try:
            connection = _read_only_connection(self.run_dir / "ledger.sqlite", self.timeout)
            rows = connection.execute("SELECT * FROM events WHERE event_id>? ORDER BY event_id",
                                      (self.last_event_id,)).fetchall()
            incremental = [{"event_id": row["event_id"], "at": row["at"], "kind": row["kind"],
                            "payload": _event_payload(row)} for row in rows]
            if rows:
                self.last_event_id = rows[-1]["event_id"]
            self._events.extend(incremental)
            snapshot = _snapshot(self.run_dir, connection, self._events, self._accepted_cache)
            snapshot["events"] = incremental
            return {"snapshot": snapshot, "events": incremental}
        except Exception as exc:  # progress must never decide the task result
            return {"snapshot": {"available": False, "error": f"{type(exc).__name__}: {exc}"}, "events": []}
        finally:
            if connection is not None:
                connection.close()

    def snapshot(self) -> dict:
        return self.poll()["snapshot"]

    def emit(self, *, now=None) -> bool:
        result = self.poll()
        snapshot = result["snapshot"]
        if not snapshot.get("available"):
            return False
        current = time.monotonic() if now is None else now
        try:
            tty = bool(self.stream.isatty())
        except (AttributeError, OSError):
            tty = False
        line = format_progress(snapshot, language=self.language)
        comparison_line = format_progress(snapshot, language=self.language, include_elapsed=tty)
        due = current - self._last_output >= 15.0
        changed = comparison_line != self._last_line
        if line and (changed or (not tty and due) or (tty and due)):
            print(line, file=self.stream, flush=True)
            self._last_line, self._last_output = comparison_line, current
            rendered = True
        else:
            rendered = False
        for event in result["events"]:
            if event["kind"] == "stage_finished":
                summary = format_stage_summary(event["payload"], language=self.language)
                if summary:
                    print(summary, file=self.stream, flush=True)
                    rendered = True
            elif event["kind"] == "round_finished":
                summary = format_round_summary(snapshot, event["payload"].get("round_index"),
                                               language=self.language,
                                               phase=event["payload"].get("phase"),
                                               event_id=event["event_id"])
                if summary:
                    print(summary, file=self.stream, flush=True)
                    rendered = True
        return rendered


def _label(language, zh, en):
    return zh if language == "zh" else en


def _stage_label(stage, language):
    labels = {"prepare": ("文献准备", "document preparation"),
              "examiner": ("文献抽取", "extraction"),
              "composite": ("候选合并", "candidate merge"),
              "embeddings": ("证据索引", "evidence indexing"),
              "cross": ("交叉核验", "cross-verification"),
              "ensemble": ("共识确认", "consensus")}
    return _label(language, *labels[stage]) if stage in labels else stage


def _money(value):
    return "?" if value is None else f"{value:.6g}"


def _round_value(value):
    return "?" if value is None else str(value)


def format_progress(snapshot: dict, *, language="en", include_elapsed=True) -> str:
    """Render one stable status line suitable for a terminal or a test stream."""
    language = language_for(language=language)
    if not isinstance(snapshot, dict) or not snapshot.get("available"):
        return ""
    phase = _label(language, "试运行", "trial") if snapshot.get("phase") == "trial" else \
        _label(language, "批处理", "batch") if snapshot.get("phase") == "batch" else "-"
    current = snapshot.get("current") or {}
    stage = _stage_label(current.get("stage") or snapshot.get("state", "-"), language)
    round_text = f"R{snapshot['round']:02d}" if isinstance(snapshot.get("round"), int) else "R-"
    scope = len(snapshot.get("scope") or [])
    extraction = snapshot.get("extraction", {})
    cross = snapshot.get("cross", {})
    wait = snapshot.get("waiting")
    wait_text = "-" if not wait else _label(language, {"model_response": "模型响应", "rate": "限速等待",
                                                         "model_loading": "模型加载"}.get(wait.get("kind"), "等待"),
                                             {"model_response": "model response", "rate": "rate wait",
                                              "model_loading": "model loading"}.get(wait.get("kind"), "waiting"))
    if wait:
        details = [str(wait["model"])] if wait.get("model") else []
        if include_elapsed and isinstance(wait.get("elapsed_seconds"), (int, float)):
            details.append(f"{wait['elapsed_seconds']:.0f}s")
        if details:
            wait_text += " " + " ".join(details)
    cost = snapshot.get("cost", {})
    cross_text = "?" if cross.get("candidates") is None else \
        f"{cross.get('completed_pairs', 0)}/{cross.get('verifier_pairs', 0)}"
    elapsed = ""
    if include_elapsed:
        if isinstance(snapshot.get("stage_elapsed_seconds"), (int, float)):
            elapsed += _label(language, f" 阶段耗时 {snapshot['stage_elapsed_seconds']:.0f}s",
                              f" stage elapsed={snapshot['stage_elapsed_seconds']:.0f}s")
        if isinstance(snapshot.get("phase_elapsed_seconds"), (int, float)):
            elapsed += _label(language, f" 阶段总耗时 {snapshot['phase_elapsed_seconds']:.0f}s",
                              f" phase elapsed={snapshot['phase_elapsed_seconds']:.0f}s")
    if language == "zh":
        return (f"阶段 {stage} {round_text} {phase} 范围{scope} | 提取 {extraction.get('completed', 0)}/{extraction.get('expected', 0)} "
                f"| 核验对 {cross_text} | 等待 {wait_text} | 缓存 {snapshot.get('cache_reuse', 0)} "
                f"| 已知成本小计 {_money(cost.get('known_cost_subtotal', cost.get('known_cost')))} "
                f"未定价请求 {cost.get('unpriced_requests', '?')}{elapsed}")
    return (f"stage {stage} {round_text} {phase} scope={scope} | extraction {extraction.get('completed', 0)}/{extraction.get('expected', 0)} "
            f"| cross pairs {cross_text} | wait {wait_text} | cache {snapshot.get('cache_reuse', 0)} "
            f"| known cost subtotal {_money(cost.get('known_cost_subtotal', cost.get('known_cost')))} "
            f"unpriced requests={cost.get('unpriced_requests', '?')}{elapsed}")


def format_round_summary(snapshot: dict, round_index: int | None = None, *, language="en",
                         phase=None, event_id=None) -> str:
    """Render paths and current dual-threshold accepted rows after a round event."""
    language = language_for(language=language)
    rounds = snapshot.get("rounds", []) if isinstance(snapshot, dict) else []
    selected = [r for r in rounds
                if (round_index is None or r.get("round_index") == round_index)
                and (phase is None or r.get("phase") == phase)
                and (event_id is None or r.get("event_id") == event_id)]
    if not selected:
        return ""
    lines = []
    for item in selected[-1:]:
        accepted = item.get("accepted", {})
        counters = item.get("counters", {})
        paths = item.get("paths", [])
        prefix = _label(language, "回合完成", "round finished")
        number = item.get("round_index", "?")
        round_text = f"R{number:02d}" if isinstance(number, int) else f"R{number}"
        phase_text = item.get("phase")
        phase_label = _label(language, "试运行", "trial") if phase_text == "trial" else \
            _label(language, "批处理", "batch") if phase_text == "batch" else ""
        line = f"{prefix} {round_text} {phase_label} | "
        line += _label(language, f"提取 {counters.get('extraction', '?')} 核验 {counters.get('cross', '?')}",
                       f"extraction={counters.get('extraction', '?')} cross={counters.get('cross', '?')}")
        budget_delta = item.get("budget_delta", {})
        new_usage = item.get("new_usage", {})
        new_requests = item.get("new_requests", budget_delta.get("calls", "?"))
        known_tokens = new_usage.get("known_tokens", budget_delta.get("known_tokens", "?"))
        known_cost = new_usage.get("known_cost", budget_delta.get("known_cost", "?"))
        unpriced = new_usage.get("unpriced_requests", budget_delta.get("unpriced_requests", "?"))
        new_requests, known_tokens, known_cost, unpriced = map(
            _round_value, (new_requests, known_tokens, known_cost, unpriced))
        if language == "zh":
            line += f" | 新请求 {new_requests} 新token {known_tokens} 新已知成本 {known_cost} 未定价 {unpriced}"
            line += (f" | 成功 {counters.get('success', '?')} 失败 {counters.get('failed', '?')}"
                     f" 未解决 {counters.get('unresolved', '?')} 缓存 {counters.get('scoped_cache_events', counters.get('cache_reuse', '?'))}")
        else:
            line += f" | new requests={new_requests} new tokens={known_tokens} new known cost={known_cost} unpriced={unpriced}"
            line += (f" | success={counters.get('success', '?')} failed={counters.get('failed', '?')}"
                     f" unresolved={counters.get('unresolved', '?')} cache={counters.get('scoped_cache_events', counters.get('cache_reuse', '?'))}")
        for label, data in accepted.items():
            rows = data.get("rows") if isinstance(data, dict) else None
            consensus = data.get("min_consensus_models") if isinstance(data, dict) else None
            if consensus is None:
                consensus = item.get("min_consensus_models")
            threshold_label = f"{label}"
            threshold_label += _label(language, f"(共识模型数≥{_round_value(consensus)})",
                                      f"(min_consensus_models={_round_value(consensus)})")
            line += f" | {threshold_label} " + (_label(language, f"接受行 {rows}", f"accepted rows={rows}") if rows is not None
                                                else _label(language, "接受行未知", "accepted rows=unknown"))
            previews = data.get("previews", []) if isinstance(data, dict) else []
            if previews:
                line += f" [{'; '.join(json.dumps(p, ensure_ascii=False, default=str) for p in previews[:5])}]"
        if paths:
            line += _label(language, " | 路径 " + ", ".join(paths), " | paths " + ", ".join(paths))
        lines.append(line)
    return "\n".join(lines)


def format_stage_summary(event: dict, *, language="en") -> str:
    """Render one completed stage and its durable output paths."""
    if not isinstance(event, dict) or not event.get("stage"):
        return ""
    language = language_for(language=language)
    stage = _stage_label(event["stage"], language)
    number = event.get("round_index")
    round_text = "" if number is None else (f" R{number:02d}" if isinstance(number, int) else f" R{number}")
    paths = [path for path in event.get("paths", []) if isinstance(path, str)]
    prefix = _label(language, "阶段完成", "stage finished")
    result = f"{prefix} {stage}{round_text}"
    if paths:
        result += _label(language, " | 路径 " + ", ".join(paths), " | paths " + ", ".join(paths))
    return result


def accepted_round_data(run_dir, spec: dict, round_index: int, *, frozen_artifacts=None) -> dict:
    """Public safe helper for callers that need structured accepted-row data."""
    try:
        return _accepted_rows(Path(run_dir).resolve(), spec, round_index, frozen_artifacts)
    except Exception as exc:  # presentation-only helper
        return {"error": f"{type(exc).__name__}: {exc}"}


__all__ = ["ProgressMonitor", "accepted_round_data", "format_progress", "format_round_summary",
           "format_stage_summary", "language_for"]
