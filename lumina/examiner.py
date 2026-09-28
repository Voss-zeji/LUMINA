from __future__ import annotations

import os
import time

import pandas as pd

from . import prompts
from .common import (
    paper_prefix_from_path,
    refineJsonString,
    save_dataframe,
    turnIntoPureText,
    write_invalid_text,
)
from .llm import single_chat
from .utils import model_name, sleep_for_rate_limit


# ---- Stage 1: Examiner — Single LLM call for one paper × one question ----
# Constructs the 3-message prompt structure:
#   msg[0] = system role + domain context
#   msg[1] = output format instruction + paper markdown content
#   msg[2] = domain-specific question (JSON schema)
# Returns (response_text, total_tokens)
def run_llm_prompt_mode(
    llm: dict,
    llm_settings: dict,
    text: str,
    paper_index,
    q_index,
    question: str,
    domain: str,
    round_index=1,
    temperature=0.01,
):
    messages = [
        {"role": "system", "content": prompts.message_system_v2.format(domain=domain)},
        {"role": "user", "content": prompts.message_system_v2_output.format(content=text)},
        {"role": "user", "content": question.strip("\n")},
    ]
    return single_chat(llm, llm_settings, messages, temperature=temperature)


# Parse the LLM's JSON response into a DataFrame
# Uses refineJsonString (json5 parser) to handle malformed JSON
# Each top-level key becomes an "item" column, sub-keys become columns
def _df_from_result(result: str) -> pd.DataFrame:
    parsed = refineJsonString(result)
    return pd.DataFrame([{"item": k, **v} for k, v in parsed.items()])


def _successful_output(output_file: str) -> bool:
    try:
        result = pd.read_csv(output_file, sep="\t")
    except Exception:
        return False
    if result.empty or "item" not in result.columns:
        return False
    if result["item"].astype(str).str.lower().isin({"ilegal", "invalid_syntax"}).any():
        return False
    return "total_tokens" in result.columns and (pd.to_numeric(result["total_tokens"], errors="coerce") >= 0).all()


# ---- Stage 1: Examiner — Main loop for a domain ----
# For each paper × each LLM × each question:
#   1. Skip if output CSV already exists (resume-safe)
#   2. Call run_llm_prompt_mode() with the truncated markdown
#   3. Parse JSON response → DataFrame
#   4. On request or parse failure: save raw text to _invalid.txt and leave the task resumable
#   5. Sleep for rate limiting (59s for low-limit, 1s for high-limit)
#   6. Save CSV with metadata columns (paper_index, model, tokens, time, question_index)
def run_examiner_for_domain(domain: str, domain_cfg: dict, llm_dicts: dict, llm_settings: dict, run_cfg: dict) -> None:
    from pathlib import Path

    markdowns = sorted(Path(domain_cfg["markdown_dir"]).glob("*.md"))
    questions = prompts.questions_for_domain(domain)

    for raw_markdown in markdowns:
        before_refs = turnIntoPureText(raw_markdown)
        paper_prefix = paper_prefix_from_path(raw_markdown)

        for llm_key, llm in llm_dicts.items():
            for q_idx, question in enumerate(questions, start=1):
                output_file = save_dataframe(llm, paper_prefix, q_idx, domain_cfg["examiner_output"], run_cfg["round_index"])
                if os.path.exists(output_file) and _successful_output(output_file):
                    print(f"Paper {paper_prefix} | {llm_key} - Question {q_idx} ... exists")
                    continue
                if os.path.exists(output_file):
                    os.remove(output_file)

                time1 = time.time()
                try:
                    result, token = run_llm_prompt_mode(
                        llm,
                        llm_settings,
                        before_refs,
                        paper_prefix,
                        q_idx,
                        question,
                        domain_cfg["domain_knowledge"],
                        round_index=run_cfg["round_index"],
                        temperature=run_cfg["temperature"],
                    )
                except Exception as exc:
                    write_invalid_text(output_file, f"LLM_ERROR: {exc}")
                    print(f"Paper {paper_prefix} | {llm_key} - Question {q_idx} ... failed: {exc}")
                    continue
                time2 = time.time()

                try:
                    result_df = _df_from_result(result)
                except Exception as exc:
                    write_invalid_text(output_file, result)
                    print(f"Paper {paper_prefix} | {llm_key} - Question {q_idx} ... invalid response: {exc}")
                    continue

                result_df["paper_index"] = paper_prefix
                result_df["total_tokens"] = token
                result_df["time_consumption"] = time2 - time1
                result_df["question_index"] = str(q_idx).zfill(2)
                result_df["model"] = model_name(llm)
                emission_type = prompts.emission_type_for_question(domain, q_idx)
                if emission_type:
                    result_df["emission_type"] = emission_type

                result_df.to_csv(output_file, sep="\t", index=False)
                Path(str(output_file).replace(".csv", "_invalid.txt")).unlink(missing_ok=True)
                sleep_for_rate_limit(llm)
                print(f"Paper {paper_prefix} | {llm_key} - Question {q_idx} ... finished")
