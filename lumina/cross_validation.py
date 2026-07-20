from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.text_splitter import MarkdownTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

from . import prompts
from .common import (
    ensure_directory_exists,
    error_cross_result,
    invalid_cross_result,
    paper_prefix_from_path,
    refineJsonString,
    turnIntoPureText,
)
from .llm import embedding_response, llm_requery
from .utils import model_name, sleep_for_rate_limit

_INVALID_EVIDENCE = {"nan", "na", "n/a", "none", "null", ""}


# ---- Stage 3: Embeddings — Generate and cache chunk vectors for one domain ----
# 1. Split each paper's markdown into chunks (chunk_size=2048, overlap=20%)
# 2. Generate embeddings via the configured embedding model
# 3. Cache as .npy files for reuse across cross-validation runs
def save_embeddings(file_path: str, array: np.ndarray) -> None:
    ensure_directory_exists(Path(file_path).parent)
    np.save(file_path, array)


def load_embeddings(file_path: str):
    if not os.path.exists(file_path):
        return None
    return np.load(file_path)


def embedding_file(domain_cfg: dict, paper_prefix: str, chunk_size: int, overlap_percent: int) -> str:
    return os.path.join(
        domain_cfg["embedding_dir"],
        f"Paper{str(paper_prefix).zfill(2)}",
        f"ChunkSize{str(chunk_size).zfill(5)}_Overlap{str(overlap_percent).zfill(3)}.npy",
    )


def _splitter(run_cfg: dict) -> MarkdownTextSplitter:
    return MarkdownTextSplitter(
        chunk_size=run_cfg["chunk_size"],
        chunk_overlap=run_cfg["chunk_size"] * (run_cfg["overlap_percent"] / 100),
    )


def generate_embeddings_for_domain(domain_cfg: dict, embedding_model: dict, llm_settings: dict, run_cfg: dict) -> None:
    splitter = _splitter(run_cfg)
    for raw_markdown in sorted(Path(domain_cfg["markdown_dir"]).glob("*.md")):
        before_refs = turnIntoPureText(raw_markdown)
        paper_prefix = paper_prefix_from_path(raw_markdown)
        out_file = embedding_file(domain_cfg, paper_prefix, run_cfg["chunk_size"], run_cfg["overlap_percent"])
        if os.path.exists(out_file):
            continue
        chunks = splitter.split_text(before_refs)
        vectors = np.array([embedding_response(chunk, embedding_model, llm_settings) for chunk in chunks])
        save_embeddings(out_file, vectors)
        print(f"embedded Paper {paper_prefix} ({len(chunks)} chunks)")


def _chunks_for_paper(markdown_path: str, run_cfg: dict) -> list[str]:
    return _splitter(run_cfg).split_text(turnIntoPureText(markdown_path))


# ---- Stage 4: Cross-Validation — Main loop for one domain ----
# For each question × each row in composite × each evidence:
#   1. Compute embedding of the evidence text
#   2. Cosine similarity against all chunk embeddings → find the best chunk
#   3. Extend context by ±text_extension chunks (default: ±1, so 3 chunks total)
#   4. For each OTHER LLM (not the one that produced the evidence):
#      a. Build the checker_requery prompt with context + evidence + key_topic
#      b. Call llm_requery() → returns {"existing_flag": 0|1, "direct_quote": "..."}
#      c. Save per-verification CSV with metadata (similarity, token, time)
#   5. aggregate_cross_scores() merges all verification votes back into composite
def cross_validate_domain(
    domain: str,
    domain_cfg: dict,
    llm_dicts: dict,
    llm_settings: dict,
    embedding_model: dict,
    run_cfg: dict,
    selected_model_names: list[str],
) -> None:
    questions = prompts.questions_for_domain(domain)
    md_by_prefix = {
        paper_prefix_from_path(p): str(p) for p in sorted(Path(domain_cfg["markdown_dir"]).glob("*.md"))
    }

    for question_index in range(1, len(questions) + 1):
        composite_file = Path(domain_cfg["composite_dir"]) / f"{domain_cfg['composite_prefix']}_Q{question_index:02d}.xlsx"
        if not composite_file.exists():
            print(f"missing composite: {composite_file}")
            continue
        composite = pd.read_excel(composite_file)

        for all_index, row in composite.iterrows():
            input_model = str(row.get("model", ""))
            if input_model not in selected_model_names:
                continue
            evidence = str(row.get("evidence", ""))
            if evidence.strip().lower() in _INVALID_EVIDENCE:
                continue

            paper_prefix = str(row.get("paper_index")).zfill(2)
            md_path = md_by_prefix.get(paper_prefix) or md_by_prefix.get(str(row.get("paper_index")))
            if md_path is None:
                continue
            chunks = _chunks_for_paper(md_path, run_cfg)
            emb_file = embedding_file(domain_cfg, paper_prefix, run_cfg["chunk_size"], run_cfg["overlap_percent"])
            response_embeddings = load_embeddings(emb_file)
            if response_embeddings is None:
                generate_embeddings_for_domain(domain_cfg, embedding_model, llm_settings, run_cfg)
                response_embeddings = load_embeddings(emb_file)

            description_embedding = np.array(embedding_response(evidence, embedding_model, llm_settings))
            similarities = [
                cosine_similarity(description_embedding.reshape(1, -1), response_embedding.reshape(1, -1))[0][0]
                for response_embedding in response_embeddings
            ]
            chunk_id = int(np.argmax(similarities))
            ext = run_cfg["text_extension"]
            optimal_context = "".join(chunks[max(0, chunk_id - ext) : chunk_id + ext + 1])

            crosser_paper_path = Path(domain_cfg["crosser_dir"]) / f"Paper_{paper_prefix}" / f"Q{question_index:02d}"
            ensure_directory_exists(crosser_paper_path)

            for llm in llm_dicts.values():
                output_model = model_name(llm)
                if output_model not in selected_model_names:
                    continue
                output_file = crosser_paper_path / f"ItemRawIndex_{str(all_index).zfill(5)}==Input_{input_model}==Output_{output_model}.csv"
                if output_file.exists():
                    continue

                time1 = time.time()
                result, token = llm_requery(
                    llm,
                    llm_settings,
                    prompts.message_system_ragQuery.strip(),
                    prompts.checker_requery.format(
                        answer=evidence,
                        context=optimal_context,
                        key_topic=row.get("item", ""),
                    ).strip(),
                    temperature=run_cfg["temperature"],
                )
                time2 = time.time()

                try:
                    result_df = pd.DataFrame([refineJsonString(result)])
                except Exception:
                    raw_path = str(output_file).replace(".csv", "_invalid.txt")
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write("" if result is None else str(result))
                    fallback = error_cross_result() if result is None else invalid_cross_result()
                    result_df = pd.DataFrame([fallback])
                    token = 0

                result_df["evaluate_token"] = token
                result_df["time_consumption"] = time2 - time1
                result_df["paper_index"] = row.get("paper_index")
                result_df["question_index"] = question_index
                result_df["item_raw_index"] = all_index
                result_df["input_model"] = input_model
                result_df["output_model"] = output_model
                result_df["max_similarities"] = float(np.nanmax(similarities))
                result_df.to_csv(output_file, sep="\t", index=False)
                sleep_for_rate_limit(llm, low_limit_seconds=5, high_limit_seconds=0.1)


# ---- Stage 4: Cross-Validation — Aggregate verification scores ----
# 1. Collect all cross-validation CSV rows
# 2. Pivot to get a flag per verifying model (existing_flag=1 → flag=1)
# 3. Sum flags across models → cross_score (how many verifying models confirm the evidence)
# 4. Merge cross_score back into the composite Excel for ensemble filtering
def aggregate_cross_scores(domain_cfg: dict, selected_model_names: list[str]) -> pd.DataFrame:
    rows = []
    for csv_file in Path(domain_cfg["crosser_dir"]).glob("Paper_*/Q*/*.csv"):
        try:
            rows.append(pd.read_csv(csv_file, sep="\t"))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()

    all_votes = pd.concat(rows, ignore_index=True)
    all_votes["flag"] = (pd.to_numeric(all_votes["existing_flag"], errors="coerce") == 1).astype(int)
    pivot = all_votes.pivot_table(
        index=["paper_index", "question_index", "item_raw_index", "input_model"],
        columns="output_model",
        values="flag",
        aggfunc="max",
        fill_value=0,
    ).reset_index()
    pivot["paper_index"] = pd.to_numeric(pivot["paper_index"], errors="coerce")

    vote_cols = [c for c in pivot.columns if c in selected_model_names]
    rename = {c: f"flag_{c}" for c in vote_cols}
    pivot = pivot.rename(columns=rename)
    flag_cols = [rename[c] for c in vote_cols]
    pivot["cross_score"] = pivot[flag_cols].sum(axis=1) if flag_cols else 0

    crosser_dir = Path(domain_cfg["crosser_dir"])
    ensure_directory_exists(crosser_dir)
    pivot.to_excel(crosser_dir / "cross_scores.xlsx", index=False)

    composite_dir = Path(domain_cfg["composite_dir"])
    prefix = domain_cfg["composite_prefix"]
    merge_cols = ["paper_index", "question_index", "item_raw_index", "input_model"]
    for question_index in sorted(pivot["question_index"].dropna().unique()):
        composite_file = composite_dir / f"{prefix}_Q{int(question_index):02d}.xlsx"
        if not composite_file.exists():
            continue
        composite = pd.read_excel(composite_file).reset_index(drop=True)
        composite["item_raw_index"] = composite.index
        composite["paper_index"] = pd.to_numeric(composite["paper_index"], errors="coerce")
        part = pivot[pivot["question_index"] == question_index][merge_cols + flag_cols + ["cross_score"]]
        merged = composite.merge(
            part,
            how="left",
            left_on=["paper_index", "question_index", "item_raw_index", "model"],
            right_on=["paper_index", "question_index", "item_raw_index", "input_model"],
        )
        if "input_model" in merged.columns:
            merged = merged.drop(columns=["input_model"])
        merged.to_excel(composite_file, index=False)
    return pivot
