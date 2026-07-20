from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import ensemble_utils_meta as eum
from . import ensemble_utils_value as euv
from . import prompts
from .common import ensure_directory_exists

_AQUA_Q1_ITEMS = ["Study_location", "Study_location_detail", "Study_period", "Latitude", "Longitude"]


def _ensemble_subset(domain: str, question_index: int, subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pd.DataFrame()
    detail = pd.DataFrame()
    if subset.empty:
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
    result = pd.DataFrame()
    detail = pd.DataFrame()
    for paper_index in sorted(df.paper_index.dropna().unique()):
        subset = df[df.paper_index == paper_index].copy()
        r, d = _ensemble_subset(domain, question_index, subset)
        result = pd.concat([result, r], axis=0)
        detail = pd.concat([detail, d], axis=0)
    ensure_directory_exists(out_dir)
    result.to_excel(out_dir / f"{domain}_Ensemble_Result_Q{question_index:02d}_{label}.xlsx", index=False)
    detail.to_excel(out_dir / f"{domain}_Ensemble_All-Standard-Answers_Q{question_index:02d}_{label}.xlsx", index=False)
    print(f"saved ensemble {domain} Q{question_index:02d} {label} ({len(result)} rows)")


def run_ensemble_for_domain(domain: str, domain_cfg: dict, run_cfg: dict) -> None:
    ensure_directory_exists(domain_cfg["ensemble_dir"])
    questions = prompts.questions_for_domain(domain)
    for question_index in range(1, len(questions) + 1):
        composite_file = Path(domain_cfg["composite_dir"]) / f"{domain_cfg['composite_prefix']}_Q{question_index:02d}.xlsx"
        if not composite_file.exists():
            continue
        df = pd.read_excel(composite_file)
        _run_variant(domain, question_index, df, Path(domain_cfg["ensemble_dir"]) / "00_full", "full")
        if "cross_score" in df.columns:
            for min_score in run_cfg.get("min_cross_scores", []):
                subset = df[df["cross_score"] >= min_score]
                if not subset.empty:
                    _run_variant(
                        domain,
                        question_index,
                        subset,
                        Path(domain_cfg["ensemble_dir"]) / f"MiniCross{min_score:02d}",
                        f"MiniCross{min_score:02d}",
                    )
