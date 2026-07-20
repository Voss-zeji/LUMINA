from __future__ import annotations

import os
import time

import pandas as pd

from . import prompts
from .common import (
    invalid_examiner_result,
    paper_prefix_from_path,
    refineJsonString,
    save_dataframe,
    turnIntoPureText,
    write_invalid_text,
)
from .llm import single_chat
from .utils import model_name, sleep_for_rate_limit


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
    try:
        return single_chat(llm, llm_settings, messages, temperature=temperature)
    except Exception:
        return (
            """
        {
          "ilegal": {
            "value": null,
            "description": null,
            "confidence_lv": null
          }
        }
        """,
            -1,
        )


def _df_from_result(result: str) -> pd.DataFrame:
    parsed = refineJsonString(result)
    return pd.DataFrame([{"item": k, **v} for k, v in parsed.items()])


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
                if os.path.exists(output_file):
                    print(f"Paper {paper_prefix} | {llm_key} - Question {q_idx} ... exists")
                    continue

                time1 = time.time()
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
                time2 = time.time()

                try:
                    result_df = _df_from_result(result)
                except Exception:
                    write_invalid_text(output_file, result)
                    result_df = pd.DataFrame([{"item": k, **v} for k, v in invalid_examiner_result().items()])

                result_df["paper_index"] = paper_prefix
                result_df["total_tokens"] = token
                result_df["time_consumption"] = time2 - time1
                result_df["question_index"] = str(q_idx).zfill(2)
                result_df["model"] = model_name(llm)
                emission_type = prompts.emission_type_for_question(domain, q_idx)
                if emission_type:
                    result_df["emission_type"] = emission_type

                result_df.to_csv(output_file, sep="\t", index=False)
                sleep_for_rate_limit(llm)
                print(f"Paper {paper_prefix} | {llm_key} - Question {q_idx} ... finished")
