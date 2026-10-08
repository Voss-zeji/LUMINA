from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ensemble_utils_meta as eum
from . import ensemble_utils_value as euv
from . import prompts
from .common import canonical_paper_id, ensure_directory_exists, read_composite, write_dataframe


# ---- Unfiltered diagnostic baseline: preserve the legacy aggregation policies ----
# Meta items (Q1 both domains, Q2 aqua):
#   → ensemble_utils_meta.ensemble_dataframe()
#   → Text normalization → voting → most common value wins
#   → Empty-value majority (>50%) → output empty/null
# Numeric items (Q3 aqua, Q2-4 wildfire):
#   → ensemble_utils_value.ensemble_numeric_dataframe()
#   → Numeric normalization → item equivalence → voting → most common wins
#   → Decimal-majority mode: cross-item aggregation by numeric value
def _ensemble_subset(domain: str, question_index: int, subset: pd.DataFrame, domain_cfg=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule = prompts.question_rule(domain, question_index, domain_cfg)
    result = pd.DataFrame()
    detail = pd.DataFrame()
    if subset.empty:
        common = ['paper_index', 'question_index', 'item']
        numeric = rule['kind'] == 'numeric'
        if numeric:
            result = pd.DataFrame(columns=common + ['ensemble_value', 'unit', 'unit_is_explicit', 'vote_count', 'models'])
            detail = pd.DataFrame(columns=common + ['normalized_value', 'scalar_value', 'unit', 'unit_is_explicit', 'count', 'confidence_sum', 'models'])
        else:
            result = pd.DataFrame(columns=common + ['ensemble_value', 'support', 'method', 'n_models'])
            detail = pd.DataFrame(columns=common + ['normalized_value', 'count', 'fraction'])
        return result, detail

    if rule['items']:
        subset = subset[subset.item.isin(rule['items'])]
    if rule['kind'] == 'numeric':
        return euv.ensemble_numeric_dataframe(subset), euv.ensemble_numeric_dataframe_all_data(subset)
    for item in rule['items'] or list(subset.item.unique()):
        items = subset[subset.item == item]
        if not items.empty:
            result = pd.concat([result, eum.ensemble_dataframe(items)], axis=0)
            detail = pd.concat([detail, eum.ensemble_dataframe_all_data(items)], axis=0)
    return result, detail


def consensus_threshold(models: list[str], run_cfg: dict) -> int:
    """Freeze the denominator before verification removes any candidates."""
    if not models or len(models) != len(set(models)):
        raise ValueError('consensus requires an explicit, unique selected model pool')
    threshold = run_cfg.get('min_consensus_models', len(models) // 2 + 1)
    if type(threshold) is not int or not 1 <= threshold <= len(models):
        raise ValueError(f'min_consensus_models must be an integer from 1 to {len(models)}')
    return threshold


def confirm_consensus(domain: str, question_index: int, frame: pd.DataFrame,
                      models: list[str], verification_threshold: int,
                      baseline_threshold: int, domain_cfg=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accept only the unique mode of verified candidates with enough source models.

    Votes belong to independent generating models, not rows or verifier flags.
    Units and explicit experiment markers delimit comparable observations.
    Existing value normalizers remain authoritative; no unit conversion is added.
    """
    rule = prompts.question_rule(domain, question_index, domain_cfg)
    numeric = rule['kind'] == 'numeric'
    result, detail = _ensemble_subset(domain, question_index, frame.iloc[:0], domain_cfg)
    extra = ['accepted', 'aggregation_mode', 'verification_threshold', 'consensus_threshold',
             'models', 'n_models', 'experimental', 'support_candidate_ids', 'gate_reason']
    result = result.reindex(columns=list(dict.fromkeys([*result.columns, *extra])))
    detail = detail.reindex(columns=list(dict.fromkeys([*detail.columns, *extra])))
    records = []
    invalid = {'', 'nan', 'na', 'n/a', 'none', 'null'}
    for row in frame.to_dict('records'):
        if str(row.get('evidence', '')).strip().lower() in invalid:
            continue
        if row.get('model') not in models:
            raise ValueError('candidate source model is outside the selected model pool')
        if not float(row['cross_score']) >= verification_threshold:
            continue
        item = str(row['item']).strip()
        if rule['items'] and item not in rule['items']:
            continue
        if numeric:
            normalized, scalar, unit, explicit = euv._normalize_value_unit_pair(row.get('value'), row.get('unit'))
            if not normalized or euv._is_meaningless_value(scalar, euv._DEFAULT_MEANINGLESS_VALUES):
                continue
        else:
            normalized = eum._normalize_single_value(row.get('value'), item)
            if normalized and eum._detect_item_category(item) == 'coord':
                normalized, _ = eum._choose_coordinate_by_precision_from_normalized_counts({normalized: 1}, 2)
            if not normalized:
                continue
            scalar, unit, explicit = normalized, '', False
        experiment = row.get('experimental', '')
        experiment = '' if pd.isna(experiment) else str(experiment).strip()
        records.append(dict(paper_index=canonical_paper_id(row['paper_index']),
                            question_index=question_index, item=item,
                            item_key=euv.item_equivalence_key(item), experimental=experiment,
                            normalized_value=normalized, scalar_value=scalar, unit=unit,
                            unit_is_explicit=explicit, model=row['model'],
                            candidate_id=str(row.get('candidate_id', ''))))
    if not records:
        return result, detail
    candidates = pd.DataFrame(records)
    accepted_rows, detail_rows = [], []
    for _, group in candidates.groupby(['paper_index', 'item_key', 'unit', 'experimental'], sort=True):
        boxes = []
        for normalized, votes in group.groupby('normalized_value', sort=True):
            source_models = sorted(set(votes.model))
            first = votes.iloc[0]
            boxes.append(dict(paper_index=first.paper_index, question_index=question_index,
                              item=sorted(set(votes.item))[0], normalized_value=normalized,
                              scalar_value=first.scalar_value, unit=first.unit,
                              unit_is_explicit=bool(votes.unit_is_explicit.any()),
                              count=len(source_models), models=','.join(source_models),
                              n_models=len(models), experimental=first.experimental,
                              support_candidate_ids='|'.join(sorted(set(votes.candidate_id) - {''})),
                              aggregation_mode='dual_threshold', verification_threshold=verification_threshold,
                              consensus_threshold=baseline_threshold))
        maximum = max(box['count'] for box in boxes)
        tied = sum(box['count'] == maximum for box in boxes) > 1
        for box in boxes:
            reason = ('tied_mode' if tied and box['count'] == maximum else
                      'not_mode' if box['count'] != maximum else
                      'below_consensus_threshold' if maximum < baseline_threshold else 'accepted')
            box.update(accepted=reason == 'accepted', gate_reason=reason,
                       fraction=box['count'] / len(models))
            detail_rows.append(box)
            if box['accepted']:
                accepted_rows.append(dict(box, ensemble_value=box['scalar_value'],
                                          vote_count=box['count'], support=box['count'], method='dual_threshold'))
    if accepted_rows:
        result = pd.DataFrame(accepted_rows).reindex(columns=result.columns)
    detail = pd.DataFrame(detail_rows).reindex(columns=detail.columns)
    return result, detail


def _run_variant(domain: str, question_index: int, df: pd.DataFrame, out_dir: Path, label: str,
                 models: list[str] | None = None, verification_threshold: int | None = None,
                 baseline_threshold: int | None = None, domain_cfg=None) -> None:
    if verification_threshold is not None:
        result, detail = confirm_consensus(domain, question_index, df, models,
                                           verification_threshold, baseline_threshold, domain_cfg)
    else:
        result, detail = _ensemble_subset(domain, question_index, df.iloc[:0], domain_cfg)
        df = df.copy()
        df['paper_index'] = df['paper_index'].map(canonical_paper_id)
        for paper_index in sorted(df.paper_index.dropna().unique()):
            subset = df[df.paper_index == paper_index].copy()
            r, d = _ensemble_subset(domain, question_index, subset, domain_cfg)
            result = pd.concat([result, r], axis=0)
            detail = pd.concat([detail, d], axis=0)
        for table in (result, detail):
            table['accepted'] = False
            table['aggregation_mode'] = 'diagnostic'
    ensure_directory_exists(out_dir)
    write_dataframe(result, out_dir / f"{domain}_Ensemble_Result_Q{question_index:02d}_{label}.xlsx")
    write_dataframe(detail, out_dir / f"{domain}_Ensemble_All-Standard-Answers_Q{question_index:02d}_{label}.xlsx")
    print(f"saved ensemble {domain} Q{question_index:02d} {label} ({len(result)} rows)")


# ---- Stage 5: Ensemble — Main entry point for a domain ----
# For each question:
#   1. Read the composite Excel (with cross_score merged from cross-validation)
#   2. Preserve diagnostic full; each MiniCross requires Tvfy AND Tbsl.
#   3. Each variant produces 2 Excel files:
#      - Ensemble_Result: accepted consensus values (full is diagnostic only)
#      - All-Standard-Answers: normalized candidates and gate decisions
def validate_cross_coverage(frame: pd.DataFrame, models: list[str] | None = None) -> None:
    if 'cross_score' not in frame:
        raise ValueError('cross_score missing; run cross before filtered ensemble')
    invalid = {'', 'nan', 'na', 'n/a', 'none', 'null'}
    eligible = ~frame['evidence'].astype(str).str.strip().str.lower().isin(invalid)
    models = models if models is not None else [c[5:] for c in frame if c.startswith('flag_')]
    if eligible.any() and len(models) < 2:
        raise ValueError('current verifier columns missing; run cross before filtered ensemble')
    scores = pd.to_numeric(frame['cross_score'], errors='coerce')
    missing = eligible & scores.isna()
    flags = []
    for model in models:
        col = f'flag_{model}'
        flag = pd.to_numeric(frame[col], errors='coerce') if col in frame else pd.Series(float('nan'), index=frame.index)
        missing |= eligible & frame['model'].ne(model) & ~flag.isin([0, 1])
        missing |= eligible & frame['model'].eq(model) & flag.fillna(0).ne(0)
        flags.append(flag.where(frame['model'].ne(model), 0).fillna(0))
    if flags:
        missing |= eligible & scores.ne(sum(flags))
    if missing.any():
        raise ValueError(f'incomplete current cross coverage: {int(missing.sum())} candidates; run cross before filtered ensemble')


def run_ensemble_for_domain(domain: str, domain_cfg: dict, run_cfg: dict,
                            selected_model_names: list[str] | None = None) -> None:
    ensure_directory_exists(domain_cfg["ensemble_dir"])
    questions = prompts.questions_for_domain(domain, domain_cfg)
    for question_index in range(1, len(questions) + 1):
        composite_file = Path(domain_cfg["composite_dir"]) / f"{domain_cfg['composite_prefix']}_Q{question_index:02d}.xlsx"
        if not composite_file.exists():
            raise FileNotFoundError(f'missing composite: {composite_file}')
        df = read_composite(composite_file)
        models = selected_model_names if selected_model_names is not None else [c[5:] for c in df if c.startswith('flag_')]
        baseline_threshold = consensus_threshold(models, run_cfg) if run_cfg.get('min_cross_scores') else None
        if run_cfg.get('min_cross_scores'):
            validate_cross_coverage(df, selected_model_names)
        _run_variant(domain, question_index, df, Path(domain_cfg["ensemble_dir"]) / "00_full", "full", domain_cfg=domain_cfg)
        for min_score in run_cfg.get("min_cross_scores", []):
            _run_variant(domain, question_index, df,
                Path(domain_cfg["ensemble_dir"]) / f"MiniCross{min_score:02d}", f"MiniCross{min_score:02d}",
                models, min_score, baseline_threshold, domain_cfg)
