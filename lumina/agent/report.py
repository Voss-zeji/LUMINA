"""Machine completion, engineering coverage and scientific acceptance stay separate.

Every number is read back from the frozen specification, the SQLite ledger and the
artifacts physically on disk.  A stage checkpoint is never trusted as proof:
artifacts are re-hashed and re-parsed with the scientific core's own validators, and
anything unreadable becomes a diagnostic instead of a quiet zero.  Nothing here
judges scientific correctness, so scientific_validation is always NOT_EVALUATED.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import pandas as pd

from .. import ensemble_utils_meta as eum
from .. import ensemble_utils_value as euv
from .. import prompts
from ..common import (filename_token, fingerprint, invalid_path, load_json, paper_markdowns,
                      paper_prefix_from_path, read_composite, save_json, turnIntoPureText)
from ..cross_validation import (_INVALID_EVIDENCE, _chunks_for_paper, _embedding_spec,
                               _valid_vectors, load_embeddings, verifier_signatures)
from ..examiner import _successful_output
from .contracts import TaskKey, file_hash
from .store import StoreError

_NO_VALUE = {"", "nan", "na", "n/a", "nat", "none", "null", "999999", "not provided",
             "not specified", "ilegal", "invalid_syntax", "[]"}
_CROSS_COLUMNS = {"candidate_id", "input_model", "output_model", "existing_flag",
                  "direct_quote", "verification_fingerprint"}


# ---------------------------------------------------------------- helpers

def _empty(value) -> bool:
    """Empty, NA, or one of the explicit 'no value' sentinels the core already knows."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NO_VALUE
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _has_evidence(value) -> bool:
    """Exactly the evidence test cross_validation uses to decide what needs verifying."""
    if _empty(value):
        return False
    return not isinstance(value, str) or value.strip().lower() not in _INVALID_EVIDENCE


def _rel(runtime, path) -> str:
    root, target = runtime.run_dir.resolve(), Path(path).resolve()
    return target.relative_to(root).as_posix() if target.is_relative_to(root) else str(target)


def _note(diagnostics, section, file, error):
    diagnostics.append(dict(section=section, file=None if file is None else str(file), error=str(error)))


def _guard(diagnostics, section, produce, fallback):
    """One broken artifact must never hide the paid run's own outcome.

    A damaged authority is the single exception: StoreError still stops the run.
    """
    try:
        return produce()
    except StoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - a diagnostic gap is not a reason to fail the run
        detail = f"{type(exc).__name__}: {exc}"
        _note(diagnostics, section, None, detail)
        return dict(fallback, status="unavailable", error=detail)


# ------------------------------------------------- frozen, secret-free identities

def _providers(spec) -> dict:
    """Non-secret provider definitions rebuilt from the frozen specification alone.

    The CLI report command has no live config and no credentials, so every endpoint
    used below is reconstructed from what this run froze.
    """
    settings = {}
    for model in spec["models"]:
        settings.setdefault(model["source"],
                           dict(url=model["endpoint"],
                                supports_json_mode=model.get("supports_json_mode", True)))
    return settings


def _signatures(spec) -> dict:
    """Current verifier signature per selected model display name."""
    embedding = dict(spec["embedding"], url=spec["embedding"].get("url") or spec["embedding"]["endpoint"])
    llm_dicts = {m["model"]: dict(model=m["model"], source=m["source"]) for m in spec["models"]}
    return verifier_signatures(llm_dicts, _providers(spec), embedding, spec["run"],
                               {'question_set': spec['question_set']} if 'question_set' in spec else None)


def _paper_index(runtime) -> dict:
    """prepared Markdown stem -> paper_uid, i.e. the paper_index the core wrote."""
    mapping = {}
    try:
        for index, path in paper_markdowns(runtime.run_dir / "outputs" / "prepared").items():
            mapping[index] = path.stem
    except (ValueError, OSError):
        pass
    for paper in runtime.spec["papers"]:
        mapping.setdefault(paper_prefix_from_path(paper["paper_uid"] + paper["suffix"]),
                           paper["paper_uid"])
    return mapping


def _index_for(index_by_uid, paper_uid) -> str:
    return next((index for index, uid in index_by_uid.items() if uid == paper_uid), paper_uid)


def _task_id(runtime, stage, paper_uid, question, **fields) -> str:
    return TaskKey(run_id=runtime.run_dir.name, stage=stage, paper_uid=paper_uid,
                   question=question, **fields).task_id


# -------------------------------------------------------------- extraction

def _examiner_file(runtime, index, question, model, round_index) -> Path:
    display = filename_token(model.split("/")[-1].lower())
    return (runtime.run_dir / "outputs" / f"R{round_index:02d}" / "examiner" / f"Paper_{index}"
            / f"{display}_P{index}_Q{question:02d}_R{round_index:02d}.csv")


def _extraction(runtime, scope, index_by_uid, prepared_fingerprints, diagnostics) -> dict:
    """Coverage over the frozen paper x question x model x round denominator.

    A valid "the paper reports no such value" answer is completed work and keeps its
    own classification; a missing, changed or unparseable output is not completed work.
    """
    spec, store = runtime.spec, runtime.store
    expected = len(scope) * len(spec["models"]) * len(spec["questions"]) * len(spec["rounds"])
    classes, answers = Counter(), Counter()
    missing, changed, failed = [], [], []

    for paper_uid in sorted(scope):
        index = _index_for(index_by_uid, paper_uid)
        for question in spec["questions"]:
            for model in spec["models"]:
                for round_index in spec["rounds"]:
                    where = dict(paper_uid=paper_uid, question=question["index"],
                                 round_index=round_index, model=model["model"])
                    task = store.task(_task_id(runtime, "examiner", paper_uid, question["index"],
                                               source_model=model["model"], round_index=round_index))
                    path = _examiner_file(runtime, index, question["index"], model["model"], round_index)
                    meta_file = path.with_suffix(".meta.json")
                    meta = load_json(str(meta_file)) if meta_file.is_file() else None
                    meta = dict(meta) if isinstance(meta, dict) else {}
                    if paper_uid in prepared_fingerprints:
                        meta.setdefault("paper_fingerprint", prepared_fingerprints[paper_uid])
                    if not path.is_file():
                        classes["missing_output"] += 1
                        missing.append(_rel(runtime, path))
                        continue
                    # Receipt first: bytes that no longer match the ledger were never this
                    # task's paid output, whatever they now parse as.
                    if task is None or not store.valid_task(task["task_id"]):
                        classes["changed_or_unreceipted_output"] += 1
                        changed.append(dict(where, file=_rel(runtime, path),
                                           detail="artifact bytes no longer match the ledger receipt"))
                        continue
                    # Then the core's own validator decides whether the paid bytes are usable.
                    if not _successful_output(str(path), meta, spec["domain"],
                                              {'question_set': spec['question_set']} if 'question_set' in spec else None):
                        classes["parse_failure"] += 1
                        failed.append(dict(where, file=_rel(runtime, path),
                                           raw=_rel(runtime, invalid_path(path))
                                           if invalid_path(path).is_file() else None))
                        continue
                    frame = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
                    valued = frame["value"].map(lambda v: not _empty(v))
                    evidenced = ~frame["evidence"].map(_empty)
                    answers["items"] += len(frame)
                    answers["with_value"] += int(valued.sum())
                    answers["no_value"] += int((~valued).sum())
                    answers["with_value_no_evidence"] += int((valued & ~evidenced).sum())
                    answers["no_value_with_evidence"] += int((~valued & evidenced).sum())
                    classes["completed_with_value" if valued.any() else "completed_no_value"] += 1

    completed = classes["completed_with_value"] + classes["completed_no_value"]
    return dict(
        expected=expected, completed=completed,
        coverage=completed / expected if expected else None,
        by_classification={k: v for k, v in sorted(classes.items())},
        answers=dict(answers),
        by_status=dict(Counter(t["status"] for t in store.tasks("examiner")
                               if t["payload"].get("paper_uid") in scope)),
        missing_outputs=missing, changed_outputs=changed, parse_failures=failed,
        not_completed=expected - completed,
        policy="the denominator is the frozen paper x question x model x round set; a valid "
               "no-value answer counts as completed and keeps its own classification",
    )


# -------------------------------------------------------------- composites

def _composites(runtime, scope, index_by_uid, diagnostics) -> dict:
    """Scope-filtered composites, keyed by (round, question).  Unreadable ones are diagnostics."""
    composites = {}
    for round_index in runtime.spec["rounds"]:
        for question in runtime.spec["questions"]:
            path = (runtime.run_dir / "outputs" / f"R{round_index:02d}" / "composite"
                    / f"LUMINA_Q{question['index']:02d}.xlsx")
            if not path.is_file():
                _note(diagnostics, "composite", _rel(runtime, path), "composite artifact missing")
                continue
            try:
                frame = read_composite(path)
            except Exception as exc:  # noqa: BLE001 - one unreadable table is a diagnostic
                _note(diagnostics, "composite", _rel(runtime, path), f"{type(exc).__name__}: {exc}")
                continue
            lookup = {_index_for(index_by_uid, uid): uid for uid in scope}
            kept = frame[frame["paper_index"].isin(lookup)].copy()
            kept["paper_uid"] = kept["paper_index"].map(lookup)
            composites[(round_index, question["index"])] = kept
    return composites


# ------------------------------------------------------------ verification

def _cross_rows(path):
    try:
        frame = pd.read_csv(path, sep="\t", keep_default_na=False,
                            dtype={"paper_index": str, "direct_quote": str})
    except Exception as exc:  # noqa: BLE001 - a malformed artifact is counted, never raised
        return None, f"{type(exc).__name__}: {exc}"
    missing = _CROSS_COLUMNS - set(frame.columns)
    if missing:
        return None, "verifier output lacks current columns " + str(sorted(missing))
    return frame.to_dict("records"), None


def _vote(record, expected_fingerprint):
    """A real 0/1 vote, or the reason this row cannot be counted as one."""
    flag = pd.to_numeric(record.get("existing_flag"), errors="coerce")
    if flag not in (0, 1):
        return None, f"existing_flag {record.get('existing_flag')!r} is not a 0/1 vote"
    if int(flag) == 1 and _empty(record.get("direct_quote")):
        return None, "positive vote without a direct quote"
    if str(record.get("verification_fingerprint")) != expected_fingerprint:
        return None, "outdated verification fingerprint"
    return int(flag), None


def _verification(runtime, scope, index_by_uid, composites, signatures, diagnostics):
    """Candidate x independent-verifier accounting, read from the current artifacts.

    Expected votes are exactly the candidates that carry real evidence, times every
    other selected model, under the current frozen signature.  Self votes, -1 errors,
    unselected models and outdated fingerprints are counted apart and never vote.
    """
    spec, store = runtime.spec, runtime.store
    full = {m["model"]: m["model"].split("/")[-1] for m in spec["models"]}
    names = {display: full_id for full_id, display in full.items()}
    lookup = {_index_for(index_by_uid, uid): uid for uid in scope}
    outcome, detail, candidates = Counter(), [], []

    for (round_index, question), frame in sorted(composites.items()):
        if frame.empty:
            continue
        eligible = frame[frame["evidence"].map(_has_evidence)]
        outcome["candidates_total"] += len(frame)
        outcome["without_evidence"] += len(frame) - len(eligible)

        # expected = every (candidate, source model, independent verifier) triple in scope
        expected, owner = set(), {}
        for _, row in eligible.iterrows():
            source = str(row["model"])
            if source not in names:
                outcome["not_selected_candidates"] += 1
                continue
            candidate = str(row["candidate_id"])
            owner[candidate] = (row.get("paper_uid"), source)
            for verifier in names:
                if verifier != source:
                    expected.add((candidate, source, verifier))
        outcome["eligible_candidates"] += len(eligible)
        outcome["expected_votes"] += len(expected)

        seen, rejected = {}, {}
        for index in sorted(lookup):
            folder = (runtime.run_dir / "outputs" / f"R{round_index:02d}" / "cross"
                      / f"Paper_{index}" / f"Q{question:02d}")
            if not folder.is_dir():
                continue
            for file in sorted(folder.glob("Candidate_*==Output_*.csv")):
                where = dict(file=_rel(runtime, file))
                records, error = _cross_rows(file)
                if records is None:
                    outcome["invalid_artifacts"] += 1
                    _note(diagnostics, "cross", where["file"], error)
                    continue
                for record in records:
                    key = (str(record["candidate_id"]), str(record["input_model"]),
                           str(record["output_model"]))
                    if key[1] not in names or key[2] not in names:
                        outcome["not_selected_artifacts"] += 1
                        continue
                    if key[1] == key[2]:
                        outcome["self_votes"] += 1
                        continue
                    if key not in expected:
                        outcome["unexpected_artifacts"] += 1
                        continue
                    if key in seen:
                        continue  # aggregate_cross_scores keeps the strongest flag per verifier
                    flag, why = _vote(record, fingerprint(dict(
                        candidate=key[0], verifier=signatures.get(key[2]))))
                    if flag is None:
                        outcome["stale_votes" if "fingerprint" in why else "invalid_artifacts"] += 1
                        rejected[key] = why
                        detail.append(where | dict(candidate_id=key[0], input_model=key[1],
                                                   output_model=key[2], reason=why))
                        continue
                    task_id = _task_id(runtime, "cross", owner[key[0]][0], question,
                                       source_model=names[key[1]], verifier_model=names[key[2]],
                                       round_index=round_index, candidate=key[0])
                    task = store.task(task_id)
                    bound = task and any((runtime.run_dir / a["path"]).resolve() == file.resolve()
                                         for a in task["artifacts"])
                    if not bound or not store.valid_task(task_id):
                        outcome["changed_artifacts"] += 1
                        rejected[key] = "cross artifact is not bound to its current ledger proof"
                        detail.append(where | dict(candidate_id=key[0], input_model=key[1],
                                                   output_model=key[2], reason=rejected[key]))
                        _note(diagnostics, "cross", where["file"], rejected[key])
                        continue
                    seen[key] = flag
                    outcome["completed_positive" if flag == 1 else "completed_negative"] += 1

        # An artifact that exists but cannot vote is counted once, above; only a genuinely
        # absent artifact falls back to the ledger's own status.
        for key in sorted(expected - set(seen) - set(rejected)):
            candidate, source, verifier = key
            paper_uid = owner[candidate][0]
            task = store.task(_task_id(runtime, "cross", paper_uid, question,
                                       source_model=names[source], verifier_model=names[verifier],
                                       round_index=round_index, candidate=candidate))
            status = task["status"] if task else "pending"
            # A succeeded receipt whose artifact vanished or changed is missing, not stale:
            # the ledger claims work the disk no longer backs.
            if status == "succeeded" and not store.valid_task(task["task_id"]):
                status = "missing_artifacts"
            outcome[{"failed": "failed_votes", "unresolved": "unresolved_votes"}.get(
                status, "missing_artifacts" if status == "missing_artifacts"
                else "stale_votes" if status == "succeeded" else "pending_votes")] += 1
            if status != "pending":
                detail.append(dict(candidate_id=candidate, input_model=source, output_model=verifier,
                                   reason=f"ledger task {status} without a usable current artifact"))

        for _, row in eligible.iterrows():
            source = str(row["model"])
            if source not in names:
                continue
            candidate = str(row["candidate_id"])
            score = pd.to_numeric(row.get("cross_score"), errors="coerce")
            candidates.append(dict(
                paper_uid=row.get("paper_uid"), question=question, round_index=round_index,
                item=row.get("item"), candidate_id=candidate, model=source,
                score=None if pd.isna(score) else int(score),
                expected_votes=len(names) - 1,
                verified=all((candidate, source, v) in seen for v in names if v != source)))

    counted = outcome["completed_positive"] + outcome["completed_negative"]
    return dict(
        eligible_candidates=outcome["eligible_candidates"],
        candidates_total=outcome["candidates_total"],
        without_evidence=outcome["without_evidence"],
        not_selected_candidates=outcome["not_selected_candidates"],
        expected_votes=outcome["expected_votes"],
        completed_positive=outcome["completed_positive"],
        completed_negative=outcome["completed_negative"],
        failed_votes=outcome["failed_votes"], pending_votes=outcome["pending_votes"],
        unresolved_votes=outcome["unresolved_votes"],
        missing_artifacts=outcome["missing_artifacts"],
        changed_artifacts=outcome["changed_artifacts"],
        invalid_artifacts=outcome["invalid_artifacts"], stale_votes=outcome["stale_votes"],
        self_votes=outcome["self_votes"], not_selected_artifacts=outcome["not_selected_artifacts"],
        unexpected_artifacts=outcome["unexpected_artifacts"],
        incomplete_candidates=sum(not c["verified"] for c in candidates),
        score_distribution={str(k): v for k, v in sorted(Counter(
            c["score"] for c in candidates if c["score"] is not None).items())},
        unscored_candidates=sum(c["score"] is None for c in candidates),
        verification_coverage=counted / outcome["expected_votes"] if outcome["expected_votes"] else None,
        detail=detail, candidates=candidates,
        policy="only candidates with real evidence, each verified by every other selected model at "
               "the current frozen signature; failed, pending, unresolved, self, unselected and "
               "outdated artifacts are counted separately and never become votes",
    )


# -------------------------------------------------------------- thresholds

def _ensemble_result_kind(path, incomplete, threshold):
    """populated, legitimate_empty, incomplete_verification, missing or unreadable."""
    if not path.is_file():
        return "missing", "threshold result artifact missing"
    if incomplete:
        # A filtered table built while votes were missing says nothing yet.
        return "incomplete_verification", f"{incomplete} candidates were not fully verified"
    try:
        frame = pd.read_excel(path)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return "unreadable", f"{type(exc).__name__}: {exc}"
    if frame.empty:
        return "legitimate_empty", "verification complete and no record passed both consensus gates" if threshold is not None else "no diagnostic records"
    return "populated", None


def _thresholds(runtime, composites, verification, diagnostics) -> dict:
    """Candidate-level pass rate over verifiable candidates; never an accuracy."""
    spec = runtime.spec
    results = {}
    for threshold in [None, *spec["run"].get("min_cross_scores", [])]:
        label = "full" if threshold is None else f"MiniCross{threshold:02d}"
        folder = "00_full" if threshold is None else label
        rows = []
        for round_index, question in sorted(composites):
            entries = [c for c in verification["candidates"]
                       if c["round_index"] == round_index and c["question"] == question]
            verifiable = [c for c in entries if c["verified"]]
            passing = [c for c in verifiable if threshold is None or (c["score"] or 0) >= threshold]
            incomplete = len(entries) - len(verifiable)
            path = (runtime.run_dir / "outputs" / f"R{round_index:02d}" / "ensemble" / folder
                    / f"{spec['domain']}_Ensemble_Result_Q{question:02d}_{label}.xlsx")
            kind, why = _ensemble_result_kind(path, incomplete, threshold)
            accepted_records = None
            if kind in {'legitimate_empty', 'populated'} and threshold is not None:
                accepted_records = len(pd.read_excel(path))
            if kind in {"missing", "unreadable", "incomplete_verification"}:
                _note(diagnostics, "ensemble", _rel(runtime, path), why)
            rows.append(dict(round_index=round_index, question=question,
                             file=_rel(runtime, path) if path.is_file() else None,
                             eligible_candidates=len(entries), verifiable_candidates=len(verifiable), passing=len(passing),
                             pass_rate=len(passing) / len(entries) if entries else None,
                             possible_upper_rate=(len(passing) + incomplete) / len(entries) if entries else None,
                             rate_status="lower_bound_incomplete" if incomplete else "complete",
                             incomplete_candidates=incomplete, result_kind=kind,
                             consensus_threshold=spec['run']['min_consensus_models'] if threshold is not None else None,
                             accepted_records=accepted_records,
                             aggregation_mode='diagnostic' if threshold is None else 'dual_threshold'))
        results[label] = rows
    return dict(thresholds=results,
                policy="pass_rate is verification support only, not final acceptance; accepted_records requires "
                       "both verification and distinct-source-model consensus. 00_full is diagnostic. "
                       "known passing / all evidence-eligible candidates; incomplete candidates never inflate "
                       "the rate, and possible_upper_rate marks unresolved support; an empty filtered result is "
                       "legitimate_empty only when verification was complete, otherwise it is "
                       "incomplete_verification; this rate is not scientific accuracy")


# -------------------------------------------------------------- embeddings

def _embeddings(runtime, scope, index_by_uid, composites, diagnostics) -> dict:
    """Evidence vectors and metadata, re-validated against the current prepared input."""
    spec, run_cfg = runtime.spec, runtime.spec["run"]
    embedding = dict(spec["embedding"], url=spec["embedding"].get("url") or spec["embedding"]["endpoint"])
    settings = _providers(spec)
    settings.setdefault(embedding["source"], dict(url=embedding["url"]))
    needed = {(row.get("paper_uid"), round_index)
              for (round_index, _), frame in composites.items()
              for _, row in frame[frame["evidence"].map(_has_evidence)].iterrows()}
    states = []
    for paper_uid in sorted(scope):
        index = _index_for(index_by_uid, paper_uid)
        prepared = runtime.run_dir / "outputs" / "prepared" / (paper_uid + ".md")
        for round_index in spec["rounds"]:
            base = (runtime.run_dir / "outputs" / f"R{round_index:02d}" / "embeddings" / f"Paper{index}"
                    / f"ChunkSize{run_cfg['chunk_size']:05d}_Overlap{run_cfg['overlap_percent']:03d}")
            matrix, meta_file = base.with_suffix(".npy"), base.with_suffix(".meta.json")
            state, detail, chunks = None, None, None
            meta = load_json(str(meta_file)) if meta_file.is_file() else None
            if not matrix.is_file() and not meta_file.is_file():
                state = "missing" if (paper_uid, round_index) in needed else "not_required"
                detail = None if state == "not_required" else "evidence vectors missing for a candidate round"
            elif not matrix.is_file() or not meta_file.is_file():
                state, detail = "metadata_missing", "matrix and metadata must both exist"
            elif not isinstance(meta, dict):
                state, detail = "metadata_missing", "embedding metadata is not a JSON object"
            elif not prepared.is_file():
                state, detail = "input_missing", "prepared input for the frozen paper is absent"
            else:
                chunks = meta.get("chunks")
                expected = fingerprint(dict(
                    spec=_embedding_spec(embedding, settings, run_cfg),
                    chunks=_chunks_for_paper(str(prepared), run_cfg)))
                vectors = load_embeddings(str(matrix))
                if meta.get("fingerprint") != expected:
                    state, detail = "stale_metadata", \
                        "embedding fingerprint differs from the current frozen inputs"
                elif not _valid_vectors(vectors, chunks):
                    state, detail = "invalid_vectors", \
                        "matrix shape, dtype, finiteness or row norms are invalid"
                elif meta.get("dimensions") != vectors.shape[1]:
                    state, detail = "invalid_vectors", "recorded dimensions differ from the stored matrix"
                else:
                    state = "valid"
            states.append(dict(paper_uid=paper_uid, round_index=round_index,
                               file=_rel(runtime, matrix), state=state, detail=detail, chunks=chunks))
            if state not in {"valid", "not_required"}:
                _note(diagnostics, "embeddings", _rel(runtime, matrix), detail)
    return dict(papers=states, states=dict(Counter(s["state"] for s in states)),
                valid=sum(s["state"] == "valid" for s in states),
                policy="a vector set counts only when its bytes, metadata fingerprint, dimensions "
                       "and the prepared input all still agree")


# --------------------------------------------------- disagreement, warnings

def _numeric(question, spec) -> bool:
    cfg = {'question_set': spec['question_set']} if 'question_set' in spec else None
    return prompts.question_rule(spec['domain'], question, cfg)['kind'] == 'numeric'


def _normalize(value, unit, item, numeric) -> str:
    """Reuse the core's own normalizers; convert no unit and pick no winner."""
    if _empty(value):
        return ""
    if numeric:
        return euv.normalize_numeric_with_explicit_unit(value, unit) or ""
    return eum._normalize_single_value(str(value), str(item),
                                       translator=lambda text: (text, None)) or ""


def _disagreement(runtime, composites) -> dict:
    """Original extractions that do not agree, preserved exactly as extracted."""
    conflicts = []
    for (round_index, question), frame in sorted(composites.items()):
        if frame.empty:
            continue
        numeric = _numeric(question, runtime.spec)
        for (paper_uid, item), group in frame.groupby(["paper_uid", "item"], dropna=False):
            answers = {str(row["model"]): dict(value=row.get("value"), unit=row.get("unit"),
                                               normalized=_normalize(row.get("value"), row.get("unit"),
                                                                     item, numeric))
                       for _, row in group.iterrows()}
            distinct = {answer["normalized"] for answer in answers.values()}
            if len(distinct) > 1:
                conflicts.append(dict(paper_uid=paper_uid, question=question, item=item,
                                      round_index=round_index, models=len(answers),
                                      distinct_values=len(distinct), answers=answers))
    return dict(conflict_count=len(conflicts), conflicts=conflicts,
                policy="disagreements come from the original extraction; the report never converts "
                       "units and never selects a winner")


def _warnings(runtime, composites) -> dict:
    """Malformed values, unsupported units and outliers, as warnings only."""
    items = []
    for (round_index, question), frame in sorted(composites.items()):
        if frame.empty or not _numeric(question, runtime.spec):
            continue
        for (paper_uid, item), group in frame.groupby(["paper_uid", "item"], dropna=False):
            scalars = {}
            for _, row in group.iterrows():
                where = dict(paper_uid=paper_uid, question=question, round_index=round_index,
                             item=item, model=row.get("model"), value=row.get("value"),
                             unit=row.get("unit"))
                if _empty(row.get("value")):
                    continue
                scalar = euv.normalize_numeric_string(row.get("value"))
                if scalar is None:
                    items.append(dict(kind="malformed_numeric_value", **where))
                    continue
                if not (euv.normalize_explicit_unit(row.get("unit")) or euv.normalize_unit(row.get("unit"))):
                    items.append(dict(kind="unsupported_unit", **where))
                scalars[str(row["model"])] = float(scalar)
            ordered = sorted(scalars.values())
            if len(ordered) < 2:
                continue
            median = ordered[len(ordered) // 2]
            for model, value in scalars.items():
                # ponytail: median-only outlier screen. A robust IQR/MAD test earns its
                # keep once real extractions show the median rule misfires.
                if value != median and abs(value - median) > 100 * max(abs(median), 1e-12):
                    items.append(dict(kind="numeric_outlier", paper_uid=paper_uid, question=question,
                                      round_index=round_index, item=item, model=model, value=value,
                                      peer_median=median))
    return dict(count=len(items), items=items,
                policy="warnings only; the control layer never rewrites a scientific value")


# ------------------------------------------------------------- accounting

def _artifacts(runtime) -> list:
    """Every production artifact with the hash of the bytes that are actually there."""
    root = runtime.run_dir / "outputs"
    if not root.is_dir():
        return []
    return [dict(path=_rel(runtime, file), sha256=file_hash(file), bytes=file.stat().st_size)
            for file in sorted(root.rglob("*")) if file.is_file()]


def _archived_qc(runtime, final_qc, diagnostics) -> dict:
    """Re-check what the archive claims instead of trusting the checkpoint."""
    if not isinstance(final_qc, dict):
        return dict(status="NOT_RUN", claimed_passed=False, outputs=[])
    states, outputs = Counter(), []
    root = (runtime.run_dir / "outputs").resolve()
    for artifact in final_qc.get("outputs", []):
        path = (runtime.run_dir / str(artifact.get("path", ""))).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            status, detail = "missing", "archived artifact is absent or outside outputs/"
        elif file_hash(path) != artifact.get("sha256"):
            status, detail = "changed", "archived artifact no longer matches its recorded hash"
        else:
            status, detail = "verified", None
        states[status] += 1
        outputs.append(dict(path=artifact.get("path"), status=status, detail=detail))
        if status != "verified":
            _note(diagnostics, "archived_qc", artifact.get("path"), detail)
    verified = states["verified"] == sum(states.values()) and bool(states)
    return dict(status="PASSED" if final_qc.get("passed") and verified else "FAILED",
                claimed_passed=bool(final_qc.get("passed")), by_status=dict(states), outputs=outputs,
                policy="the checkpoint's own claim is re-verified against the bytes on disk; a "
                       "claim that no longer matches those bytes is reported as FAILED")


def _outcome(state, pending, archived) -> str:
    if state == "FATAL_ERROR":
        return "failure"
    if state == "DONE" and not pending and archived["status"] == "PASSED":
        return "success"
    return "incomplete"


# ------------------------------------------------------------------ public

def write_report(runtime, name="final_report", *, paper_uids=None, final_qc=None):
    spec, store = runtime.spec, runtime.store
    scope = set(paper_uids or [p["paper_uid"] for p in spec["papers"]])
    snapshot, totals = store.snapshot(), runtime.budget.totals()
    events = store._events()
    diagnostics = []
    expected = len(scope) * len(spec["models"]) * len(spec["questions"]) * len(spec["rounds"])

    index_by_uid = _guard(diagnostics, "paper_index", lambda: _paper_index(runtime), {})
    prepared_fingerprints = _guard(diagnostics, "prepared_inputs", lambda: {
        paper["paper_uid"]: fingerprint(turnIntoPureText(runtime.run_dir / "outputs" / "prepared"
                                                         / (paper["paper_uid"] + ".md")))
        for paper in spec["papers"] if paper["paper_uid"] in scope
        and (runtime.run_dir / "outputs" / "prepared" / (paper["paper_uid"] + ".md")).is_file()}, {})
    preparation_records = []
    for paper in spec['papers']:
        metadata = runtime.run_dir / 'outputs' / 'prepared' / (paper['paper_uid'] + '.meta.json')
        if paper['paper_uid'] in scope and metadata.is_file():
            record = load_json(metadata)
            if isinstance(record, dict):
                preparation_records.append(record)
    signatures = _guard(diagnostics, "verifier_signatures", lambda: _signatures(spec), {})
    composites = _guard(diagnostics, "composites",
                        lambda: _composites(runtime, scope, index_by_uid, diagnostics), {})
    extraction = _guard(diagnostics, "extraction",
                        lambda: _extraction(runtime, scope, index_by_uid, prepared_fingerprints,
                                            diagnostics),
                        dict(expected=expected, completed=0, coverage=None))
    verification = _guard(diagnostics, "verification",
                          lambda: _verification(runtime, scope, index_by_uid, composites,
                                                signatures, diagnostics),
                          dict(eligible_candidates=0, expected_votes=0, candidates=[],
                               completed_positive=0, completed_negative=0))
    thresholds = _guard(diagnostics, "thresholds",
                        lambda: _thresholds(runtime, composites, verification, diagnostics),
                        dict(thresholds={}))
    embeddings = _guard(diagnostics, "embeddings",
                        lambda: _embeddings(runtime, scope, index_by_uid, composites, diagnostics),
                        dict(papers=[], states={}, valid=0))
    disagreement = _guard(diagnostics, "disagreement",
                          lambda: _disagreement(runtime, composites), dict(conflict_count=0, conflicts=[]))
    warnings = _guard(diagnostics, "warnings", lambda: _warnings(runtime, composites),
                      dict(count=0, items=[]))
    artifacts = _guard(diagnostics, "artifacts", lambda: _artifacts(runtime), [])
    archived = _guard(diagnostics, "archived_qc",
                      lambda: _archived_qc(runtime, final_qc, diagnostics),
                      dict(status="NOT_RUN", claimed_passed=False, outputs=[]))

    errors = [e for e in events if e["kind"].startswith("error") or e["payload"].get("error")]
    pending = [g for g in store.gates() if g["status"] == "pending"]
    state = snapshot["run"]["state"]
    result = dict(
        run_id=runtime.run_dir.name, state=state,
        outcome=_outcome(state, pending, archived),
        scientific_validation="NOT_EVALUATED", domain=spec["domain"],
        paper_count=len(scope), model_count=len(spec["models"]),
        question_count=len(spec["questions"]), rounds=spec["rounds"],
        specification_hash=snapshot["run"]["spec_hash"],
        inputs=[{k: p[k] for k in ("paper_uid", "sha256", "display_name")}
                for p in spec["papers"] if p["paper_uid"] in scope],
        models=spec["models"], scientific_code=spec["scientific_code"], budget=totals,
        preparation=preparation_records,
        budget_limits=spec["budget"],
        pricing_basis={m: p["basis"] for m, p in spec["pricing"].items()},
        extraction=extraction, verification=verification, thresholds=thresholds,
        embeddings=embeddings, disagreement=disagreement, warnings=warnings,
        diagnostics=diagnostics, outputs=artifacts, artifacts=artifacts,
        archived_qc=archived, final_qc=final_qc or dict(passed=False, status="NOT_RUN"),
        time=dict(started_at=snapshot["run"]["started_at"], updated_at=snapshot["run"]["updated_at"],
                  runtime_seconds=totals.get("runtime"), generated_at=time.time()),
        calls=dict(attempts=totals.get("attempts"), calls=totals.get("calls"),
                   open_slots=totals.get("open_slots"),
                   retries=sum(e["kind"] == "request_rejected" for e in events),
                   unknown_usage=totals.get("unknown_usage"),
                   unknown_requests=totals.get("unknown_requests"), imports=totals.get("imports")),
        tokens=dict(known=totals.get("known_tokens"), held=totals.get("held_tokens")),
        costs=dict(known=totals.get("known_cost"), held=totals.get("held_cost")),
        cache_hits=sum(e["kind"] in {"cache_hit", "response_reused"} for e in events),
        imports=totals.get("imports"),
        fingerprints=dict(specification=snapshot["run"]["spec_hash"],
                          inputs={p["paper_uid"]: p["sha256"] for p in spec["papers"]
                                  if p["paper_uid"] in scope},
                          scientific_code=spec["scientific_code"],
                          control_code=spec.get("control_code", {})),
        task_counts=snapshot["tasks"], pending_gates=[g["gate_id"] for g in pending], errors=errors,
        runtime_policy="elapsed wall clock since run creation; waiting and pauses do not reset "
                       "the limit",
        verification_policy=spec["policies"]["cross"],
    )
    save_json(runtime.run_dir / "reports" / f"{name}.json", result)
    save_json(runtime.run_dir / "metrics.json",
              dict(extraction=result["extraction"], verification=result["verification"],
                   thresholds=result["thresholds"], budget=result["budget"],
                   scientific_validation="NOT_EVALUATED"))
    store.export_views()
    return result
