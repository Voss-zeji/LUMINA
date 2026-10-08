from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (candidate_id, canonical_paper_id, invalid_path, load_json,
                     validate_answers, write_dataframe)
from .examiner import _df_from_result


def _parse_invalid_txt(invalid_txt_path: str) -> pd.DataFrame:
    with open(invalid_txt_path, encoding="utf-8") as f:
        raw = f.read()
    try:
        return _df_from_result(raw)
    except Exception:
        return pd.DataFrame()


def create_baseline_composite(domain_cfg: dict, questions: list[int], models: list[str] | None = None,
                              round_index: int = 1, domain: str | None = None,
                              expected_tasks: dict | None = None) -> dict[int, pd.DataFrame]:
    examiner_output = Path(domain_cfg["examiner_output"])
    composite_dir = Path(domain_cfg["composite_dir"])
    results: dict[int, pd.DataFrame] = {}
    columns = ["paper_index", "question_index", "round_index", "model", "item", "value", "evidence",
               "confidence_lv", "request_fingerprint", "paper_fingerprint", "candidate_id"]
    failures = []
    if expected_tasks is None:
        expected_tasks = {}
        for file in sorted(examiner_output.glob(f"Paper_*/*_R{round_index:02d}.meta.json")):
            meta = load_json(file)
            if isinstance(meta, dict) and (models is None or meta.get("model") in models):
                expected_tasks[str(file.with_name(file.name.replace(".meta.json", ".csv")))] = meta
        if not expected_tasks and any(examiner_output.glob(f"Paper_*/*_R{round_index:02d}.csv")):
            raise ValueError("legacy examiner outputs lack provenance; rerun examiner before composite")
    for question_index in questions:
        parts = []
        for output, meta in expected_tasks.items():
            if meta["question_index"] != question_index or meta["round_index"] != round_index:
                continue
            if models is not None and meta["model"] not in models:
                continue
            csv_file = Path(output)
            part = pd.DataFrame()
            try:
                if csv_file.exists():
                    part = pd.read_csv(csv_file, sep="\t", keep_default_na=False, dtype={"paper_index": str})
                    if not all(key in part and part[key].astype(str).eq(str(value)).all()
                               for key, value in meta.items()):
                        part = pd.DataFrame()
                if part.empty and invalid_path(output).exists():
                    stored = load_json(csv_file.with_suffix(".meta.json"))
                    if stored == meta:
                        part = _parse_invalid_txt(str(invalid_path(output)))
                        for key, value in meta.items():
                            part[key] = value
                validate_answers(part, domain, question_index, domain_cfg)
                part["paper_index"] = part["paper_index"].map(canonical_paper_id)
                part["question_index"] = question_index
                part["round_index"] = round_index
                part["candidate_id"] = part.apply(candidate_id, axis=1)
                parts.append(part)
            except Exception as exc:
                failures.append(f"{csv_file.name}: {type(exc).__name__}: {exc}")
        composite_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        if not composite_df.empty:
            composite_df = composite_df.drop_duplicates("candidate_id").reset_index(drop=True)
        results[question_index] = composite_df
    if failures:
        raise RuntimeError("composite failed: " + "; ".join(failures))
    # Validate the entire selected batch before publishing any replacement table.
    for question_index, frame in results.items():
        output_file = composite_dir / f"{domain_cfg['composite_prefix']}_Q{question_index:02d}.xlsx"
        write_dataframe(frame, output_file)
        print(f"saved {output_file} ({len(frame)} rows)")
    return results
