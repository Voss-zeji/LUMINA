"""Independent READ-ONLY scientific evaluation (G7).

This module never writes into a production run directory and never guesses a match
it cannot justify. It reuses the scientific core's own numeric and unit
normalisation so evaluation agrees with production without re-implementing the
policy:

* values are compared exactly, with no unit conversion and no tolerance;
* a numeric row without an explicit unit, or without experiment identity, is
  UNMATCHED rather than wrong;
* item identity is the scientific core's own equivalence class, so a raw
  ``CH4_flux`` label and the published ``CH4 flux`` label are the same item;
* experiment identity is joined from the composite rows that actually
  contributed to a published value, and a value pooled from more than one
  experiment stays UNMATCHED instead of being assigned one of them;
* candidate-set (decimal-majority) rows are scored as sets, not as one winner,
  and the mode is derived per paper x question x round x variant from exactly
  the raw subset that variant consumed;
* correct absences, extraction misses, parse failures and policy-filtered values
  are separate diagnostic classes, and only a reference ``not_reported`` row is a
  *known* absence: a prediction earns a true negative only by recording that same
  known absence, while a diagnostic prediction or a missing production record at that
  identity is reported as unresolved and excluded from the truth denominators;
* without an independent human reference the result is NOT_EVALUATED with null
  metrics, never accuracy 0.

``diagnostics``, ``unaligned`` and the false-positive / false-negative counters
are overlapping views of the same comparison, not additive totals: a filtered
value is both a miss against the reference and a diagnostic, and a mismatch is
reported once on each side deliberately. Only the reference-driven denominators
(``comparable_gold``, ``unaligned_gold``) define the metric denominator.

Model agreement is never treated as truth: the reference protocol is carried
through to the report unchanged.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .common import canonical_paper_id, save_json
from .ensemble_utils_value import (_DEFAULT_MEANINGLESS_VALUES, _is_meaningless_value,
                                   _normalize_value_unit_pair, item_equivalence_key,
                                   normalize_explicit_unit, normalize_numeric_string)
from . import prompts

EVALUATOR_VERSION = "1.2"

#: Reference provenance is reported, never upgraded. Only human protocols count as
#: scientific truth; the synthetic protocol proves the software path only.
PROTOCOLS = {
    "single_human": "one human reference read",
    "independent_double_read": "two independent human reads",
    "adjudicated": "independent double read plus documented adjudication",
    "synthetic": "synthetic fixture; software acceptance only",
}

#: Absence classes that must never be collapsed into each other or into "wrong".
DIAGNOSTIC_STATUSES = ("missing_extraction", "not_reported", "parse_failure", "filtered_by_policy")

#: The ONLY status that asserts a *known* absence of a value. A reference row carrying
#: this status is a ground-truth absence and may be scored (it defines the
#: ``not_reported_true_absence`` denominator). Every other status is a *diagnostic* about
#: the pipeline, not evidence the value is absent, so a reference row carrying one is an
#: *unresolved* reference: it is excluded from the truth denominators entirely and can
#: never become a true negative.
KNOWN_ABSENCE_STATUS = "not_reported"

_MISSING_TEXT = {"", "none", "nan", "null", "na", "n/a", "nat"}


def _sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _blank(value: Any) -> bool:
    """True for the absence spellings the production core already treats as empty."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _MISSING_TEXT


def _text(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def _check_protocol(protocol: Any) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown reference protocol {protocol!r}; expected one of {sorted(PROTOCOLS)}")
    return protocol


def _check_status(status: Any) -> Optional[str]:
    if _blank(status):
        return None
    if status not in DIAGNOSTIC_STATUSES:
        raise ValueError(f"unknown record status {status!r}; expected one of {list(DIAGNOSTIC_STATUSES)}")
    return status


def _record(raw: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Validate and normalise one evaluation record.

    Item granularity is preserved exactly: two records only ever match when their
    item strings are equal, so an original cross-item joined label never silently
    merges a finer-grained gold row.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{side} record must be an object")
    for field in ("paper_uid", "question", "item"):
        if field not in raw or _blank(raw.get(field)):
            raise ValueError(f"{side} record requires {field}")
    question, variant, round_index = raw.get("question"), raw.get("variant"), raw.get("round")
    for name, value in (("question", question), ("round", round_index)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{side} record requires a positive integer {name}")
    kind = raw.get("kind") or ("text" if _numeric_text(raw.get("value")) is None else "numeric")
    if kind not in ("numeric", "text"):
        raise ValueError(f"{side} record kind must be numeric or text")
    return {
        "kind": kind,
        "paper_uid": canonical_paper_id(raw["paper_uid"]),
        "question": question,
        "item": str(raw["item"]).strip(),
        # The core's own item equivalence class, so matching agrees with the
        # ensemble instead of comparing raw label spellings.
        "item_key": item_equivalence_key(raw["item"]),
        "variant": _text(variant),
        "round": round_index,
        "experiment_id": _text(raw.get("experiment_id")),
        "unit": normalize_explicit_unit(raw.get("unit")),
        "value": raw.get("value"),
        "status": _check_status(raw.get("status")),
        "method": _text(raw.get("method")) or "winner",
        "candidate_set": raw.get("candidate_set"),
        "raw_candidates": list(raw.get("raw_candidates") or []),
        # Production adapter evidence: the item labels and experiment identities
        # of the composite rows that actually contributed to this value.
        "source_items": [_text(v) for v in (raw.get("source_items") or [])],
        "source_experiments": [_text(v) for v in (raw.get("source_experiments") or [])],
    }


def _numeric_text(value: Any) -> Optional[str]:
    """Normalised numeric form, or None when the value is not a legitimate number."""
    if _blank(value):
        return None
    return normalize_numeric_string(value)


def _key(record: Dict[str, Any]) -> tuple:
    """Identity for matching: item equivalence class, experiment, variant, round.

    The item slot is the scientific core's own ``item_equivalence_key`` so a raw
    ``CH4_flux`` label and the published ``CH4 flux`` label are one item, exactly
    as the ensemble itself groups them. Exact-string matching would silently mark
    every real production row unmatched.
    """
    return _identity_key(record)


def _identity_key(record: Dict[str, Any]) -> tuple:
    """Paper, question, item class, variant, round and experiment identity.

    One definition for both roles it plays: the matching key for rows that assert a
    value, and the identity an absence is compared on.
    """
    return (record["paper_uid"], record["question"], record["item_key"],
            record["variant"], record["round"], record["experiment_id"])


def _context(record: Dict[str, Any]) -> tuple:
    """Grouping for per-context reporting; excludes experiment so contexts stay visible."""
    return (record["paper_uid"], record["question"], record["item_key"],
            record["variant"], record["round"])


def _absent_key(record: Dict[str, Any]) -> Optional[tuple]:
    """A row that asserts a *known* absence, so it matches on identity alone.

    Only ``not_reported`` says "this value is known to be absent". A row tagged
    ``parse_failure`` / ``missing_extraction`` / ``filtered_by_policy`` documents a
    *pipeline* outcome, not a scientific fact, so it is not a known absence: it must
    never be rewarded as a true negative and, on the reference side, it is an
    unresolved reference excluded from the truth denominators (see
    ``_unresolved_reference``). A reference that merely failed to parse or that was
    filtered cannot make a diagnostic prediction look scientifically correct.
    """
    if record["status"] != KNOWN_ABSENCE_STATUS:
        return None
    return _identity_key(record)


def _unresolved_reference(record: Dict[str, Any]) -> bool:
    """A reference row whose status is diagnostic, not a known absence.

    ``parse_failure``, ``missing_extraction`` and ``filtered_by_policy`` on the
    *reference* side mean the reference itself is unresolved about this item, so it
    cannot serve as truth for a true negative. Such rows are reported with their reason
    and excluded from the truth denominators rather than silently treated as absence.
    """
    return record["status"] is not None and record["status"] != KNOWN_ABSENCE_STATUS


def _observed_values(record: Dict[str, Any]) -> List[tuple]:
    """Comparable (value, unit) pairs this row asserts.

    A candidate set asserts every member; a winner asserts exactly one.
    """
    if record["method"] == "candidate_set" and record["candidate_set"]:
        pairs = []
        for member in record["candidate_set"]:
            if not isinstance(member, dict):
                raise ValueError("candidate_set members must be objects with value and unit")
            numeric = _numeric_text(member.get("value"))
            if numeric is None:
                continue
            pairs.append((numeric, normalize_explicit_unit(member.get("unit"))))
        return pairs
    if record["method"] == "candidate_set":
        return []
    numeric = _numeric_text(record["value"])
    if numeric is None:
        return []
    return [(numeric, record["unit"])]


def _is_malformed(record: Dict[str, Any]) -> bool:
    """A non-blank value that the core parser cannot read is a parse failure, not absence."""
    if record["kind"] == "text":
        return False
    # An explicitly declared status asserts no value at all, so whatever text the
    # row carries cannot be a failed parse of an assertion.
    if record["status"] is not None:
        return False
    return not _blank(record["value"]) and _numeric_text(record["value"]) is None


def _uncomparable(record: Dict[str, Any]) -> Optional[str]:
    """Reason this record cannot be compared at all, or None when it can.

    Every branch here is a deliberate refusal to guess. A missing explicit unit or
    missing experiment identity makes the row UNMATCHED; it never becomes an error
    and never becomes a match.

    A declared status of any kind means the row asserts no value, so it is compared
    on identity alone: requiring a unit or an experiment for a row that published
    nothing would manufacture a false negative out of a correct absence. Whether that
    non-assertion is a *known* absence or merely a pipeline outcome is decided later,
    against the reference, by ``_absent_key`` and ``_unresolved_reference``.
    """
    if record["kind"] != "numeric":
        return None
    if record["status"] is not None:
        return None
    if not record["unit"]:
        return "missing_explicit_unit"
    if not record["experiment_id"]:
        return "missing_experiment_identity"
    return None


def _assertions(record: Dict[str, Any]) -> set:
    if record["kind"] == "text":
        text = _text(record["value"])
        return {text} if text else set()
    return set(_observed_values(record))


def _case(entry: Dict[str, Any], reason: str, **extra) -> Dict[str, Any]:
    case = dict(reason=reason, paper_uid=entry["paper_uid"], question=entry["question"],
                item=entry["item"], variant=entry["variant"], round=entry["round"],
                experiment_id=entry["experiment_id"], **extra)
    # An UNMATCHED row is only actionable if it says which evidence it could not
    # choose between, so the contributing items and experiments travel with the case.
    for field in ("source_items", "source_experiments"):
        if entry.get(field):
            case[field] = entry[field]
    return case


def _candidate_strings(entry: Dict[str, Any]) -> List[str]:
    """Every value this production row could have carried, winners and raw candidates."""
    values = [str(v) for v in _observed_values(entry)]
    values.extend(str(v) for v in entry["raw_candidates"])
    return values


def _raw_candidate_pairs(entries: List[str]) -> set:
    """Parse raw candidate strings into comparable (value, unit) pairs."""
    pairs = set()
    for candidate in entries:
        joined, _, remainder = candidate.partition(" ")
        try:
            normalized, scalar, unit, _ = _normalize_value_unit_pair(joined, remainder or None)
        except Exception:
            continue
        if normalized is not None:
            pairs.add((scalar, unit))
    return pairs


def _filtered_evidence_any(records: Iterable[Dict[str, Any]], assertions: set) -> Optional[str]:
    """A raw candidate string that equals one of the reference values.

    This is what separates "the policy filtered a correct value" from "the value was
    never extracted": both look like absence in the published workbook. Only a numeric
    assertion is matched here, because raw candidate evidence is the core's numeric
    vote box; a text metadata row has no such table to search. The test is the *type* of
    the assertion, not its length: a two-character text value such as ``"up"`` would
    otherwise unpack into a fake ``("u", "p")`` value/unit pair and let a real numeric
    candidate be reported as a filtered value it never matched.
    """
    numeric_assertions = {a for a in assertions if isinstance(a, tuple) and len(a) == 2}
    if not numeric_assertions:
        return None
    for entry in records:
        pairs = _raw_candidate_pairs(_candidate_strings(entry))
        for numeric, unit in numeric_assertions:
            if (numeric, unit) in pairs:
                return next(c for c in _candidate_strings(entry)
                            if normalize_numeric_string(c) == numeric)
    return None


def _assessment(protocol: Optional[str]) -> Dict[str, Any]:
    """Protocol scope. Synthetic references are explicitly not scientific truth."""
    if protocol is None:
        return dict(protocol=None, description=None, is_scientific_truth=False,
                    limitation="no independent human reference was supplied; "
                               "machine completion and model agreement are not scientific validation")
    return dict(protocol=protocol, description=PROTOCOLS[protocol],
                is_scientific_truth=protocol != "synthetic",
                limitation=("synthetic fixture: verifies the evaluator software only"
                            if protocol == "synthetic" else
                            "single human read without independent second reader or adjudication"
                            if protocol == "single_human" else None))


def evaluate_records(predictions: List[Dict[str, Any]], gold: List[Dict[str, Any]],
                     protocol: str = "synthetic") -> Dict[str, Any]:
    """Score prediction records against reference records.

    Matching is on paper, question, item, variant, round and experiment identity.
    Metrics are computed only over rows that are genuinely comparable; unmatched
    rows are reported separately so they can never inflate a denominator.
    """
    _check_protocol(protocol)
    predicted = [_record(row, "prediction") for row in predictions]
    reference = [_record(row, "reference") for row in gold]

    unaligned: List[Dict[str, Any]] = []
    unaligned_gold: List[Dict[str, Any]] = []
    notes: List[str] = []
    diagnostics: Counter = Counter()
    # Declared before ``_index`` so the reference pass can record a gold row it refuses
    # to score; every gold row then lands in exactly one outcome bucket.
    gold_outcomes: Counter = Counter()

    # Build comparable indexes, refusing to compare what cannot be compared. A record
    # lands in exactly one bucket, so nothing is diagnosed or counted twice.
    def _index(records, side):
        by_value, by_absence, by_diagnostic = {}, {}, {}
        for entry in records:
            if _is_malformed(entry):
                diagnostics["parse_failure"] += 1
                if side == "reference":
                    # A reference row that could not be read states no fact, so it is
                    # excluded from the truth denominators on the same footing as a
                    # diagnostic-status reference.
                    gold_outcomes["unmatched"] += 1
                (unaligned if side == "prediction" else unaligned_gold).append(
                    _case(entry, "malformed_reference_value" if side == "reference" else "malformed_prediction_value",
                          value=_text(entry["value"])))
                continue
            reason = _uncomparable(entry)
            if reason:
                if side == "reference":
                    unaligned_gold.append(_case(entry, reason))
                else:
                    unaligned.append(_case(entry, reason))
                if reason == "missing_experiment_identity":
                    notes.append("numeric rows need experiment identity (composite 'experimental' column); "
                                 "an unmatched row is never assigned a gold row by item name alone")
                continue
            if side == "reference" and _unresolved_reference(entry):
                # A reference tagged with a *diagnostic* status is unresolved about this
                # item, not a known absence. It is reported with its reason and kept out
                # of the truth denominators entirely; it can never become a true negative.
                gold_outcomes["unmatched"] += 1
                unaligned_gold.append(_case(
                    entry, "unresolved_reference_status", status=entry["status"],
                    detail="reference status is a pipeline diagnostic, not a known absence; "
                           "excluded from truth denominators"))
                continue
            if entry["status"] == KNOWN_ABSENCE_STATUS:
                # A declared known absence on either side: compared on identity alone.
                by_absence.setdefault(_absent_key(entry), []).append(entry)
            elif entry["status"] is not None:
                # Prediction side: a diagnostic status means the row asserts no value, so
                # it is compared on identity, but the verdict later differs from a known
                # absence because it records a pipeline outcome rather than a fact.
                by_diagnostic.setdefault(_identity_key(entry), []).append(entry)
            else:
                # Comparable identity, no value and no declared status: the producer
                # reached this row. Either it asserts a value or it reported nothing,
                # and the reference pass decides which.
                by_value.setdefault(_key(entry), []).append(entry)
        return by_value, by_absence, by_diagnostic

    predicted_by_value, predicted_absent, predicted_diagnostic = _index(predicted, "prediction")
    reference_by_value, reference_absent, _ = _index(reference, "reference")

    # Declared statuses are diagnostics of the *output side*. They are recorded once,
    # when the row is first classified, instead of once per code path that touches it.
    for entry in predicted:
        if entry["status"] and not _is_malformed(entry) and not _uncomparable(entry):
            diagnostics[entry["status"]] += 1

    # Every prediction grouped by context, including the ones that cannot be compared
    # on their own, so a reference row can tell "nothing was produced here" apart from
    # "what was produced cannot be compared".
    predicted_by_context: Dict[tuple, List[Dict[str, Any]]] = {}
    for entry in predicted:
        predicted_by_context.setdefault(_context(entry), []).append(entry)

    true_positive = false_positive = false_negative = true_negative = 0
    per_context: Dict[tuple, Dict[str, Any]] = {}
    consumed_contexts = set()

    def _context_entry(entry, method):
        slot = per_context.setdefault(_context(entry), dict(paper_uid=entry["paper_uid"], question=entry["question"],
                                                         item=entry["item"], variant=entry["variant"],
                                                         round=entry["round"], method=method, matches=0))
        return slot

    # Reference rows first: each one decides whether it was found at all.
    for key, entries in reference_by_value.items():
        candidates = predicted_by_value.get(key)
        reference_assertions = set().union(*(_assertions(e) for e in entries))
        if candidates:
            consumed_contexts.add(_context(entries[0]))
            predicted_assertions = set().union(*(_assertions(e) for e in candidates))
            slot = _context_entry(entries[0], candidates[0]["method"])
            if reference_assertions & predicted_assertions:
                true_positive += 1
                gold_outcomes["comparable"] += len(entries)
                slot["matches"] += 1
                # A candidate set is scored as a set, so a value it carries that the
                # reference does not is not an error; only a single winner is exclusive.
                for assertion in reference_assertions - predicted_assertions:
                    present = assertion in _raw_candidate_pairs(candidates[0]["raw_candidates"])
                    unaligned.append(_case(entries[0], "assertion_not_in_output",
                                           assertion=list(assertion),
                                           classification="filtered_by_policy" if present else "missing_extraction"))
                continue
            # Present at the same identity but wrong: one false negative for the
            # reference and one false positive for the asserted value. The correct
            # value may still sit among raw candidates, which is recorded as evidence
            # rather than folded into a missing-extraction count.
            false_negative += len(entries)
            false_positive += len(candidates)
            gold_outcomes["comparable"] += len(entries)
            evidence = _filtered_evidence_any(candidates, reference_assertions)
            unaligned.append(_case(entries[0], "value_mismatch",
                                   reference_value=sorted(map(str, reference_assertions)),
                                   predicted_value=sorted(map(str, predicted_assertions)),
                                   classification="filtered_by_policy" if evidence else None,
                                   raw_candidate_evidence=evidence))
            continue
        declared = predicted_absent.get(key) or predicted_diagnostic.get(key)
        if declared:
            # The identity was reached and production published no value. The reference
            # expected one, so this is a real miss on a comparable row whichever kind of
            # non-assertion production recorded: a known absence, a failed parse, an
            # extraction miss or a policy filter are all still "no value published".
            _context_entry(entries[0], declared[0]["status"] or "not_published")
            false_negative += len(entries)
            gold_outcomes["comparable"] += len(entries)
            consumed_contexts.add(_context(entries[0]))
            unaligned.append(_case(entries[0], "reference_value_not_published",
                                   reference_value=sorted(map(str, reference_assertions))))
            continue
        siblings = predicted_by_context.get(_context(entries[0]), [])
        malformed = [s for s in siblings if _is_malformed(s)]
        if malformed:
            # The identity is fully usable; only the value could not be parsed. That is
            # a miss on a comparable row, distinct from an unusable identity.
            _context_entry(entries[0], "unparsable_value")
            false_negative += len(entries)
            gold_outcomes["comparable"] += len(entries)
            consumed_contexts.add(_context(entries[0]))
            unaligned.append(_case(entries[0], "reference_value_not_published",
                                   reference_value=sorted(map(str, reference_assertions)),
                                   value=_text(malformed[0]["value"])))
            continue
        if siblings:
            # A row exists for this context but cannot be compared: it carries no
            # explicit unit, no experiment identity, or an unparsable value. That is
            # reported as UNMATCHED rather than guessed at, and the reference row keeps
            # its own identity in the unmatched bucket.
            _context_entry(entries[0], "unmatched")
            gold_outcomes["unmatched"] += len(entries)
            unaligned.append(_case(entries[0], "prediction_not_comparable",
                                   reference_value=sorted(map(str, reference_assertions)),
                                   prediction_reasons=sorted({
                                       _uncomparable(s) or "malformed"
                                       for s in siblings})))
            continue
        _context_entry(entries[0], "no_prediction")
        gold_outcomes["unmatched"] += len(entries)
        raw_evidence = _filtered_evidence_any(predicted, reference_assertions)
        unaligned_gold.append(_case(entries[0], "no_comparable_prediction",
                                    reference_value=sorted(map(str, reference_assertions)),
                                    raw_candidate_evidence=raw_evidence))

    # Only *known* reference absences (gold ``not_reported``) need a verdict here. A
    # prediction that published nothing at an identity the reference covers as a value
    # was already counted as a miss by the pass above; one at an identity the reference
    # never mentions asserts nothing and so cannot be a false positive.
    #
    # A gold ``not_reported`` row defines a known absence, so it is the only thing that
    # can support a true negative. It is a true negative ONLY when production reached
    # the identity and legitimately recorded the same known absence (``not_reported``).
    # Every other prediction at that identity is NOT the confirmed absence:
    #   * a diagnostic status (parse_failure / missing_extraction / filtered_by_policy)
    #     means production failed or filtered, which is *unknown*, not a rewardable
    #     absence of assertion, so it is recorded as unresolved and excluded from the
    #     denominators rather than rewarded as a true negative or charged as a false one;
    #   * no production record at all means the value was simply never produced, which is
    #     a missing extraction, not a known scientific absence, so it is unmatched.
    for key, gold_entries in reference_absent.items():
        pred_absent = key in predicted_absent
        pred_entries = predicted_absent.get(key, [])
        diagnostic = predicted_diagnostic.get(key, [])
        entry = (gold_entries or pred_entries or diagnostic)[0]
        consumed_contexts.add(_context(entry))
        asserted = predicted_by_value.get(key, [])
        if pred_absent:
            true_negative += 1
            gold_outcomes["comparable"] += len(gold_entries)
        elif asserted:
            # A reference that reports no value cannot make an asserted value correct.
            false_positive += len(asserted)
            gold_outcomes["comparable"] += len(gold_entries)
            unaligned.append(_case(entry, "asserted_but_reference_not_reported"))
        elif diagnostic:
            # Production declared a diagnostic outcome at a known-absent identity. That
            # is a pipeline failure, not a confirmed scientific absence, so it is neither
            # rewarded as a true negative nor charged against accuracy. It is recorded
            # with its status and excluded from the truth denominators.
            _context_entry(entry, diagnostic[0]["status"])
            gold_outcomes["unmatched"] += len(gold_entries)
            unaligned.append(_case(
                entry, "reference_not_reported_prediction_diagnostic",
                prediction_status=diagnostic[0]["status"],
                detail="a diagnostic prediction is unknown, not a confirmed absence; "
                       "excluded from truth denominators"))
        else:
            # The reference reports a known absence and production published nothing at
            # all here. A missing prediction is NOT a confirmed scientific absence, so it
            # is reported as a missing extraction / unmatched, kept out of the accuracy
            # denominators, rather than scored as a false negative.
            _context_entry(entry, "missing_extraction")
            gold_outcomes["unmatched"] += len(gold_entries)
            unaligned.append(_case(
                entry, "reference_not_reported_prediction_missing",
                detail="no production record at a known-absent identity; treated as a "
                       "missing extraction, not a known absence, and excluded from truth denominators"))

    # Prediction rows that asserted a value no reference row covers. Rows already
    # consumed by a reference context are excluded so nothing counts twice.
    for key, entries in predicted_by_value.items():
        if key in reference_by_value or _context(entries[0]) in consumed_contexts:
            continue
        for entry in entries:
            false_positive += 1
            _context_entry(entry, entry["method"])
            unaligned.append(_case(entry, "prediction_without_reference",
                                   predicted_value=sorted(map(str, _assertions(entry)))))

    # A value the reference expected that was not published is either a policy filter
    # (it sat among raw candidates) or an extraction miss. Derived from the recorded
    # evidence, never from a bare absence, so a correct absence is never counted here.
    for entry in unaligned_gold + unaligned:
        classification = entry.get("classification") or entry.get("reason")
        if classification in ("filtered_by_policy", "missing_extraction"):
            diagnostics[classification] += 1

    counts = dict(prediction_records=len(predicted), reference_records=len(reference),
                  comparable_gold=gold_outcomes["comparable"],
                  unmatched_gold=gold_outcomes["unmatched"],
                  # Denominator for the true-negative verdict: only reference rows whose
                  # status is the KNOWN absence ``not_reported``. Diagnostic-status
                  # references and absent predictions are counted elsewhere and never
                  # enter it, so a pipeline failure can be read apart from agreement
                  # with a known reference absence.
                  not_reported_true_absence=len(reference_absent),
                  unaligned_gold=len(unaligned_gold))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else None
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else None
    metrics = dict(true_positive=true_positive, false_positive=false_positive,
                   false_negative=false_negative, true_negative=true_negative,
                   exact_match_precision=precision, exact_match_recall=recall)
    return {
        "status": "EVALUATED",
        "metrics": metrics,
        "counts": counts,
        "diagnostics": {name: diagnostics.get(name, 0) for name in DIAGNOSTIC_STATUSES},
        "per_context": list(per_context.values()),
        "unaligned": unaligned,
        "unaligned_gold": unaligned_gold,
        "unaligned_notes": sorted(set(notes)),
        "assessment": _assessment(protocol),
    }


# ---- Production adapter -----------------------------------------------------

_RESULT_RE = re.compile(r"_Ensemble_Result_Q(?P<question>\d+)_(?P<label>.+)\.xlsx$")
_COMPOSITE_RE = re.compile(r"_Q(?P<question>\d+)\.xlsx$")
_ROUND_RE = re.compile(r"^R(?P<round>\d+)$")


def _output_dir(run_dir: Path, output_dir: Optional[str]) -> Path:
    """Resolve the evaluation directory, refusing anything inside the production run.

    Both sides are resolved so a symlink cannot smuggle the output back inside the
    run directory under a different name. An *ancestor* of the run is refused too:
    writing the report into a parent would drop ``evaluation_report.json`` into a
    directory the run itself lives in and pollute the surrounding production root.
    """
    run = Path(run_dir).resolve()
    target = Path(output_dir).resolve() if output_dir else run.parent / "evaluations" / run.name
    if _conflicts_with_run(run, target):
        raise ValueError("evaluation output must be outside the production run directory "
                         "and must not be an ancestor of it")
    return target


def _conflicts_with_run(run: Path, target: Path) -> bool:
    """Whether writing to ``target`` would touch the production run's own tree.

    Three cases are refusals: the run itself, anything inside it (a direct or
    symlinked path), and any ancestor of the run, which would drop the report into a
    directory production lives in. A *sibling* under a shared parent is the normal,
    allowed case, so the comparison is on ancestry rather than on the parent alone.
    """
    return target == run or target.is_relative_to(run) or run.is_relative_to(target)


def _assert_writable_outside(run: Path, target: Path) -> None:
    """Re-check the destination immediately before every write.

    The path was validated once at the start of the run, but a junction or symlink
    can be planted at the output path while evaluation is in flight. Resolving the
    target again right before the write is what actually keeps the report out of the
    production tree; the target itself is resolved rather than its parent, so a
    legitimate sibling output directory is not rejected along with it.
    """
    resolved = target.resolve()
    if _conflicts_with_run(run, resolved):
        raise ValueError(f"evaluation output resolves inside the production run: {resolved}")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _production_files(run: Path) -> List[Dict[str, Any]]:
    """Hash every production file, so the report can prove it changed nothing.

    The evaluator never writes any of these, so the proof covers the whole run
    rather than only the outputs it happens to read: the ledger, the copied
    inputs, the checkpoints and stage artifacts, every production output, the
    manifest, state, events, errors and metrics. Hashing a subset would leave the
    rest of the run unproven.
    """
    return [dict(path=str(path.relative_to(run)), sha256=_sha256_file(path),
                 size_bytes=path.stat().st_size)
            for path in sorted(p for p in run.rglob("*") if p.is_file())]


def _composite_rows(run: Path) -> List[Dict[str, Any]]:
    """Every raw composite row, tagged with the question taken from its filename.

    The composite is the only production file that keeps the "experimental" column,
    and it is also the input the ensemble itself was computed from. Both the
    experiment identity and the decimal-majority mode therefore have to come from
    here rather than being inferred from the published workbook.
    """
    rows: List[Dict[str, Any]] = []
    for path in sorted((run / "outputs").rglob("composite/LUMINA_Q*.xlsx")):
        match = _COMPOSITE_RE.search(path.name)
        if not match:
            continue
        question = int(match.group("question"))
        frame = pd.read_excel(path, keep_default_na=False, dtype={"paper_index": str})
        if not {"paper_index", "item"}.issubset(frame.columns):
            continue
        round_index = _round_of(path)
        for row in frame.to_dict("records"):
            rows.append(dict(row, _question=question, _round=round_index))
    return rows


def _numeric_domain_question(domain: str, question: int) -> bool:
    """Whether the core ensembles this domain question numerically.

    Mirrors ``ensemble._ensemble_subset`` exactly: aqua Q1/Q2 and wildfire Q1 go
    through the text (meta) ensemble, everything else through the numeric one.
    Parsing a text row as a number would score an unparsable failure instead of
    the metadata value the ensemble actually published.
    """
    d = (domain or "").lower()
    return question != 1 and not (d == "aqua" and question == 2)


def _variant_threshold(variant: str) -> Optional[float]:
    """The ``cross_score`` floor a filtered variant consumed, or None for full.

    ``run_ensemble_for_domain`` builds each MiniCross variant by filtering
    ``cross_score >= min_score`` *before* ensembling, so reproducing its
    decimal-majority decision requires replaying that same filter.
    """
    match = re.fullmatch(r"MiniCross(?P<score>\d+)", variant or "")
    return float(match.group("score")) if match else None


def _contributing_rows(rows: List[Dict[str, Any]], paper_uid: str, question: int,
                       threshold: Optional[float]) -> List[Dict[str, Any]]:
    """The exact raw subset one paper x question variant consumed.

    ``_run_variant`` ensembles one paper at a time, so the decimal-majority
    decision must be taken over that same per-paper subset, never over the whole
    composite and never across rounds or variants.
    """
    subset = [r for r in rows
              if canonical_paper_id(r.get("paper_index")) == paper_uid
              and r["_question"] == question]
    if threshold is None:
        return subset
    kept = []
    for row in subset:
        score = row.get("cross_score")
        if score is None or _blank(score):
            continue
        try:
            if float(score) >= threshold:
                kept.append(row)
        except (TypeError, ValueError):
            continue
    return kept


def _decimal_majority(values: Iterable[Any]) -> bool:
    """Reproduce the core's decimal-majority rule on one consumed raw subset.

    ``ensemble_numeric_dataframe`` computes this over the values it was handed,
    so the evaluator applies the identical predicate to the identical rows: a
    ``>=`` comparison against 0.5 over valid, non-meaningless normalised values.
    No rounding is introduced and there is no winner fallback.
    """
    normalized = [normalize_numeric_string(v) for v in values]
    valid = [n for n in normalized
             if n is not None and not _is_meaningless_value(n, _DEFAULT_MEANINGLESS_VALUES)]
    if not valid:
        return False
    two = ["." in str(n) and len(str(n).split(".", 1)[1]) >= 2 for n in valid]
    return sum(two) / float(len(valid)) >= 0.5


def _truthy(value: Any) -> bool:
    """The producer's boolean flag as written in the Result workbook."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        return bool(value) and not pd.isna(value)
    except (TypeError, ValueError):
        return False


def _published_pair(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """(scalar, unit, core normalised value) for one published Result row.

    ``ensemble_numeric_dataframe`` writes the bare scalar plus a separate unit
    column when the unit came from the producer's own column, and the combined
    "value unit" string when it was inline. Both spellings are unpacked with the
    core's own normalisers so the value can be looked up in the same vote box
    production used. Units are compared exactly: nothing is converted.
    """
    unit = normalize_explicit_unit(row.get("unit"))
    ensemble_value = row.get("ensemble_value")
    scalar = _numeric_text(ensemble_value) or ""
    if _truthy(row.get("unit_is_explicit")):
        return scalar, unit, (f"{scalar} {unit}" if scalar and unit else scalar)
    return scalar, unit, _text(ensemble_value)


def _split_core_value(normalized: str) -> Dict[str, str]:
    """Unpack a core ``normalized_value`` back into a comparable value/unit pair.

    ``_normalize_value_unit_pair`` builds the key as ``f"{base} {unit}"``, so
    splitting on the first space is the exact inverse and needs no parsing. The unit
    is returned as-is: the core already normalised it, and no conversion happens.
    """
    scalar, _, unit = _text(normalized).partition(" ")
    return dict(value=scalar, unit=unit)


def _contribution_index(contributing: List[Dict[str, Any]]):
    """Index the composite rows that produced each core vote box.

    ``ensemble_numeric_dataframe_all_data`` groups by
    ``(paper, question, item_equiv, normalized_value)``; ``ensemble_numeric_dataframe``
    then either picks a winner inside one item equivalence class or pools a value
    across classes. Both groupings are rebuilt here with the core's own
    ``item_equivalence_key`` and ``_normalize_value_unit_pair``, so the lookup uses
    exactly the boxes production used and never a re-implemented policy.
    """
    by_class_value: Dict[tuple, List[Dict[str, Any]]] = {}
    by_value: Dict[str, List[Dict[str, Any]]] = {}
    for row in contributing:
        normalized = _normalize_value_unit_pair(row.get("value"), row.get("unit"))[0]
        if not normalized:
            continue
        by_class_value.setdefault((item_equivalence_key(row.get("item")), normalized), []).append(row)
        by_value.setdefault(normalized, []).append(row)
    return by_class_value, by_value


def _identity(witnesses: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """The single experiment behind a published value, or "" when it is not one.

    Identity comes only from the composite rows that actually contributed this
    value. When they do not agree on one explicit ``experimental`` value the answer
    is "" and the scorer reports the row UNMATCHED: a value pooled from two
    experiments belongs to no single one of them, and the published item label is
    only a representative of a class, never evidence of an experiment.
    """
    identities = {_text(r.get("experimental")) for r in witnesses}
    identities.discard("")
    if len(identities) != 1:
        return "", sorted(identities)
    (only,) = identities
    return only, sorted(identities)


def _result_rows(path: Path, round_index: int, variant: str, domain: str,
                 composite_rows: List[Dict[str, Any]]):
    """Build evaluation records from one ensemble Result workbook.

    Whether a row is numeric or text follows the domain and question, exactly as
    ``ensemble._ensemble_subset`` routes it: aqua Q1/Q2 and wildfire Q1 publish
    text metadata and must be scored as text, everything else publishes numeric
    values scored under the numeric unit/experiment rules.

    The scoring mode and the experiment identity of every row are replayed from the
    same raw composite subset this exact variant consumed, so the evaluator cannot
    disagree with production about what it published.
    """
    frame = pd.read_excel(path, keep_default_na=False, dtype={"paper_index": str})
    numeric = _numeric_domain_question(domain, int(
        _RESULT_RE.search(path.name).group("question")))
    threshold = _variant_threshold(variant)
    round_rows = [r for r in composite_rows if r["_round"] == round_index]
    # The core takes its mode decision once per paper x question, so the verdict and
    # the contribution index are cached on that granularity instead of being
    # recomputed for every published row.
    verdicts: Dict[tuple, tuple] = {}
    records: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        paper_uid = canonical_paper_id(row.get("paper_index"))
        question = int(row.get("question_index"))
        item = str(row.get("item")).strip()
        contributing = _contributing_rows(round_rows, paper_uid, question, threshold)
        if numeric:
            cache_key = (paper_uid, question)
            if cache_key not in verdicts:
                verdicts[cache_key] = (_decimal_majority(r.get("value") for r in contributing),
                                       _contribution_index(contributing))
            candidate_set_mode, (by_class_value, by_value) = verdicts[cache_key]

            _, unit, normalized = _published_pair(row)
            # The published label is only a representative of one equivalence class, so
            # identity comes from the composite rows that actually voted for this exact
            # value. Decimal-majority mode pools a value across classes, which is why
            # that mode consults the cross-class index and winner mode the class-scoped
            # one; the item label never selects an experiment on its own.
            witnesses = (by_value.get(normalized, []) if candidate_set_mode
                         else by_class_value.get((item_equivalence_key(item), normalized), []))
            experiment_id, identities = _identity(witnesses)

            # Every vote box this context produced, i.e. exactly what the core writes
            # to its All-Standard-Answers table. This is the evidence that separates a
            # value the policy filtered from one that was never extracted, so it covers
            # all candidates rather than only the published one.
            boxes = sorted({_text(v) for (cls, v) in by_class_value if cls and _text(v)})
            method, candidate_set = "winner", None
            if candidate_set_mode:
                # Decimal-majority mode publishes one row per candidate value and is
                # scored as a set, so a value it carries that the reference does not
                # is not an error.
                method = "candidate_set"
                candidate_set = [_split_core_value(v) for v in boxes] or None
            records.append(dict(
                kind="numeric", paper_uid=paper_uid, question=question, item=item,
                variant=variant, round=round_index, experiment_id=experiment_id,
                unit=unit, value=row.get("ensemble_value"),
                method=method, candidate_set=candidate_set, raw_candidates=boxes,
                source_items=sorted({str(r.get("item")).strip() for r in witnesses}),
                source_experiments=identities))
        else:
            # Text metadata: no unit and no experiment identity, matched as published.
            records.append(dict(
                kind="text", paper_uid=paper_uid, question=question, item=item,
                variant=variant, round=round_index, experiment_id="", unit="",
                value=row.get("ensemble_value"), method=_text(row.get("method")) or "text_vote",
                candidate_set=None, raw_candidates=[],
                source_items=sorted({str(r.get("item")).strip() for r in contributing
                                     if item_equivalence_key(r.get("item")) == item_equivalence_key(item)}),
                source_experiments=[]))
    return records


def _load_gold(path: str) -> Dict[str, Any]:
    """Load and validate the reference file before any production data is scored."""
    gold_path = Path(path)
    if not gold_path.is_file():
        raise FileNotFoundError(f"reference file not found: {gold_path}")
    data = _read_json(gold_path)
    if not isinstance(data, dict):
        raise ValueError("reference must be a JSON object")
    _check_protocol(data.get("protocol"))
    provenance = data.get("reference_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("reference requires a non-empty reference_provenance")
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("reference requires a records list")
    data["_path"] = gold_path
    return data


def _manifest_domain(manifest: Dict[str, Any]) -> str:
    """The run's scientific domain, refused rather than guessed.

    The core routes each question to the text or the numeric ensemble purely by domain
    and question index, so an unknown domain does not merely lose information: it sends
    aqua Q2 down the numeric branch and turns real ``Study_species`` text into an
    unparsable value. Defaulting a missing domain to ``""`` produced exactly that.

    Only the run manifest is an authority here. Inferred from file names it would be a
    guess about a partial run, so evaluation refuses instead. The core's own
    ``prompts.questions_for_domain`` validates the name, so the accepted set is never
    duplicated or widened here. This runs before anything is read or written.
    """
    specification = manifest.get("specification")
    domain = _text(specification.get("domain")) if isinstance(specification, dict) else ""
    if not domain:
        raise ValueError("run manifest has no specification.domain; the domain decides "
                         "whether a question publishes text or numeric values, so it is "
                         "refused rather than guessed from output file names")
    try:
        prompts.questions_for_domain(domain)
    except ValueError as exc:
        raise ValueError(f"unsupported run domain {domain!r}: {exc}") from exc
    return domain


def _discover(run: Path, domain: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Read production outputs read-only and report whether the run looks complete."""
    composite_rows = _composite_rows(run)
    records: List[Dict[str, Any]] = []
    for result_path in sorted((run / "outputs").rglob("ensemble/*/*_Ensemble_Result_Q*.xlsx")):
        match = _RESULT_RE.search(result_path.name)
        if not match:
            continue
        variant = result_path.parent.name
        round_index = _round_of(result_path)
        records.extend(_result_rows(result_path, round_index, variant, domain, composite_rows))
    composites = list((run / "outputs").rglob("composite/LUMINA_Q*.xlsx"))
    status = "PRODUCTION_READ" if records and composites else "INCOMPLETE_PRODUCTION"
    return records, composites, status


def _round_of(path: Path) -> int:
    for part in path.parts:
        match = _ROUND_RE.match(part)
        if match:
            return int(match.group("round"))
    raise ValueError(f"production output is not inside an R<round> directory: {path}")


def evaluate_run(run_dir, gold: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate a production run against an independent reference, read-only.

    Only the run manifest, state, production outputs and raw composites are read.
    Nothing is written inside the run directory, and no runtime API, mutating store
    or configuration is opened. Without a reference the result is NOT_EVALUATED with
    null metrics rather than accuracy 0.
    """
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory not found: {run}")
    target = _output_dir(run, output_dir)
    manifest = _read_json(run / "manifest.json") if (run / "manifest.json").is_file() else {}
    # Resolved before any production file is read or any output path is created, so a
    # run whose manifest does not state a usable domain fails without touching either.
    domain = _manifest_domain(manifest)

    before = _production_files(run)
    records, composites, production_status = _discover(run, domain)
    state = _read_json(run / "state.json") if (run / "state.json").is_file() else {}

    result: Dict[str, Any] = {
        "run_id": run.name,
        "output_dir": str(target),
        "production_status": production_status,
        "production_records": len(records),
        "composite_files": len(composites),
        "run_state": state.get("run", {}).get("state"),
        "spec_hash": state.get("run", {}).get("spec_hash"),
    }

    if gold is None:
        result.update(status="NOT_EVALUATED", metrics=None, counts=None, diagnostics=None,
                      per_context=[], unaligned=[], unaligned_gold=[], unaligned_notes=[],
                      assessment=_assessment(None),
                      scientific_validation="NOT_EVALUATED",
                      provenance=_provenance(run, before, None, manifest))
    else:
        reference = _load_gold(gold)
        scored = evaluate_records(records, reference["records"], reference["protocol"])
        result.update(scored)
        result["scientific_validation"] = (
            "SYNTHETIC_SOFTWARE_CHECK" if reference["protocol"] == "synthetic"
            else "AGREEMENT_WITH_" + reference["protocol"].upper())
        result["provenance"] = _provenance(run, before, reference, manifest)

    after = _production_files(run)
    if [f["sha256"] for f in before] != [f["sha256"] for f in after]:
        raise RuntimeError("evaluation must not modify production files")
    result["production_unchanged"] = True

    target.mkdir(parents=True, exist_ok=True)
    # Re-checked after mkdir and again before each write: a junction planted during
    # the run must not redirect a report into the production tree.
    _assert_writable_outside(run, target)
    save_json(target / "evaluation_report.json", result)
    _assert_writable_outside(run, target)
    save_json(target / "production_hashes_after.json", dict(files=after))
    return result


def _provenance(run: Path, production: List[Dict[str, Any]], reference, manifest) -> Dict[str, Any]:
    """Everything needed to re-run this evaluation against the same inputs."""
    return dict(
        run_id=run.name,
        evaluator_version=EVALUATOR_VERSION,
        evaluator_sha256=_sha256_file(Path(__file__).resolve()),
        gold_path=str(reference["_path"]) if reference else None,
        gold_sha256=_sha256_file(reference["_path"]) if reference else None,
        gold_protocol=reference["protocol"] if reference else None,
        reference_provenance=reference["reference_provenance"] if reference else None,
        specification_hash=manifest.get("specification_hash"),
        specification_version=manifest.get("format_version"),
        production_files=production,
    )
