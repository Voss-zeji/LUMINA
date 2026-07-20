from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import ensure_directory_exists, invalid_examiner_result, refineJsonString


def _parse_invalid_txt(invalid_txt_path: str) -> pd.DataFrame:
    with open(invalid_txt_path, encoding="utf-8") as f:
        raw = f.read()
    try:
        parsed = refineJsonString(raw)
        return pd.DataFrame([{"item": k, **v} for k, v in parsed.items()])
    except Exception:
        return pd.DataFrame([{"item": k, **v} for k, v in invalid_examiner_result().items()])


def create_baseline_composite(domain_cfg: dict, questions: list[int], models: list[str] | None = None) -> dict[int, pd.DataFrame]:
    examiner_output = Path(domain_cfg["examiner_output"])
    composite_dir = Path(domain_cfg["composite_dir"])
    ensure_directory_exists(composite_dir)
    results: dict[int, pd.DataFrame] = {}

    for question_index in questions:
        composite_df = pd.DataFrame()
        for paper_dir in sorted(examiner_output.glob("Paper_*")):
            for csv_file in sorted(paper_dir.glob("*.csv")):
                if f"_Q{question_index:02d}_" not in csv_file.name:
                    continue
                try:
                    part = pd.read_csv(csv_file, sep="\t")
                except Exception:
                    continue
                if models and "model" in part.columns:
                    part = part[part["model"].isin(models)]
                composite_df = pd.concat([composite_df, part], ignore_index=True)

            for invalid_file in sorted(paper_dir.glob("*_invalid.txt")):
                if f"_Q{question_index:02d}_" not in invalid_file.name:
                    continue
                part = _parse_invalid_txt(str(invalid_file))
                stem = invalid_file.name.replace("_invalid.txt", "")
                bits = stem.split("_")
                part["model"] = bits[0]
                part["paper_index"] = paper_dir.name.replace("Paper_", "")
                part["question_index"] = str(question_index).zfill(2)
                if models:
                    part = part[part["model"].isin(models)]
                composite_df = pd.concat([composite_df, part], ignore_index=True)

        if not composite_df.empty:
            output_file = composite_dir / f"{domain_cfg['composite_prefix']}_Q{question_index:02d}.xlsx"
            composite_df.to_excel(output_file, index=False)
            results[question_index] = composite_df
            print(f"saved {output_file} ({len(composite_df)} rows)")
        else:
            print(f"no rows for Q{question_index:02d}")
    return results
