from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ensemble_utils_meta as eum
from . import ensemble_utils_value as euv
from . import prompts
from .common import canonical_paper_id, ensure_directory_exists, read_composite, write_dataframe

_AQUA_Q1_ITEMS = ["Study_location", "Study_location_detail", "Study_period", "Latitude", "Longitude"]


# ---- Stage 5: Ensemble — Route to meta or numeric ensemble by question type ----
# Meta items (Q1 both domains, Q2 aqua):
#   → ensemble_utils_meta.ensemble_dataframe()
#   → Text normalization → voting → most common value wins
#   → Empty-value majority (>50%) → output empty/null
# Numeric items (Q3 aqua, Q2-4 wildfire):
#   → ensemble_utils_value.ensemble_numeric_dataframe()
#   → Numeric normalization → item equivalence → voting → most common wins
#   → Decimal-majority mode: cross-item aggregation by numeric value
def _ensemble_subset(domain: str, question_index: int, subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.DataFrame()
    detail = pd.DataFrame()
    if subset.empty:
        common = ['paper_index', 'question_index', 'item']
        numeric = question_index != 1 and not (domain == 'aqua' and question_index == 2)
        if numeric:
            result = pd.DataFrame(columns=common + ['ensemble_value', 'unit', 'unit_is_explicit', 'vote_count', 'models'])
            detail = pd.DataFrame(columns=common + ['normalized_value', 'scalar_value', 'unit', 'unit_is_explicit', 'count', 'confidence_sum', 'models'])
        else:
            result = pd.DataFrame(columns=common + ['ensemble_value', 'support', 'method', 'n_models'])
            detail = pd.DataFrame(columns=common + ['normalized_value', 'count', 'fraction'])
        return result, detail

    if domain == "aqua":
        if question_index == 1:
            for item_content in _AQUA_Q1_ITEMS:
                items = subset[subset.item == item_content]
                if items.empty:
                    continue
                r = eum.ensemble_dataframe(items)
                d = eum.ensemble_dataframe_all_data(items)
                result = pd.concat([result, r], axis=0)
                detail = pd.concat([detail, d], axis=0)
        elif question_index == 2:
            items = subset[subset.item == "Specie"]
            if not items.empty:
                result = eum.ensemble_dataframe(items)
                detail = eum.ensemble_dataframe_all_data(items)
        else:
            result = euv.ensemble_numeric_dataframe(subset)
            detail = euv.ensemble_numeric_dataframe_all_data(subset)
        return result, detail

    if question_index == 1:
        for item_content in _AQUA_Q1_ITEMS:
            items = subset[subset.item == item_content]
            if items.empty:
                continue
            r = eum.ensemble_dataframe(items)
            d = eum.ensemble_dataframe_all_data(items)
            result = pd.concat([result, r], axis=0)
            detail = pd.concat([detail, d], axis=0)
    else:
        result = euv.ensemble_numeric_dataframe(subset)
        detail = euv.ensemble_numeric_dataframe_all_data(subset)
    return result, detail


def _run_variant(domain: str, question_index: int, df: pd.DataFrame, out_dir: Path, label: str) -> None:
    result, detail = _ensemble_subset(domain, question_index, df.iloc[:0])
    df = df.copy()
    df['paper_index'] = df['paper_index'].map(canonical_paper_id)
    for paper_index in sorted(df.paper_index.dropna().unique()):
        subset = df[df.paper_index == paper_index].copy()
        r, d = _ensemble_subset(domain, question_index, subset)
        result = pd.concat([result, r], axis=0)
        detail = pd.concat([detail, d], axis=0)
    ensure_directory_exists(out_dir)
    write_dataframe(result, out_dir / f"{domain}_Ensemble_Result_Q{question_index:02d}_{label}.xlsx")
    write_dataframe(detail, out_dir / f"{domain}_Ensemble_All-Standard-Answers_Q{question_index:02d}_{label}.xlsx")
    print(f"saved ensemble {domain} Q{question_index:02d} {label} ({len(result)} rows)")


# ---- Stage 5: Ensemble — Main entry point for a domain ----
# For each question:
#   1. Read the composite Excel (with cross_score merged from cross-validation)
#   2. Run full and each configured reachable cross-score threshold.
#   3. Each variant produces 2 Excel files:
#      - Ensemble_Result:     final consensus values
#      - All-Standard-Answers: all normalized candidates with frequency counts
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
    questions = prompts.questions_for_domain(domain)
    for question_index in range(1, len(questions) + 1):
        composite_file = Path(domain_cfg["composite_dir"]) / f"{domain_cfg['composite_prefix']}_Q{question_index:02d}.xlsx"
        if not composite_file.exists():
            raise FileNotFoundError(f'missing composite: {composite_file}')
        df = read_composite(composite_file)
        if run_cfg.get('min_cross_scores'):
            validate_cross_coverage(df, selected_model_names)
        _run_variant(domain, question_index, df, Path(domain_cfg["ensemble_dir"]) / "00_full", "full")
        for min_score in run_cfg.get("min_cross_scores", []):
            subset = df[pd.to_numeric(df["cross_score"], errors='raise') >= min_score]
            _run_variant(domain, question_index, subset,
                Path(domain_cfg["ensemble_dir"]) / f"MiniCross{min_score:02d}", f"MiniCross{min_score:02d}")
