from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from . import prompts
from .common import (
    fingerprint,
    error_details,
    invalid_path,
    paper_markdowns,
    refineJsonString,
    save_json,
    save_dataframe,
    turnIntoPureText,
    validate_answers,
    write_dataframe,
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
    if not parsed or not all(isinstance(v, dict) for v in parsed.values()):
        raise ValueError("extraction must contain nonempty item objects")
    return pd.DataFrame([{"item": k, **v} for k, v in parsed.items()])


def _successful_output(output_file: str, metadata: dict | None = None, domain: str | None = None) -> bool:
    try:
        result = pd.read_csv(output_file, sep="\t", keep_default_na=False, dtype={'paper_index': str})
        validate_answers(result, domain, metadata["question_index"] if metadata else None)
        if metadata:
            return all(key in result and result[key].astype(str).eq(str(value)).all()
                       for key, value in metadata.items())
    except Exception:
        return False
    return "request_fingerprint" in result.columns


def domain_tasks(domain: str, domain_cfg: dict, llm_dicts: dict, llm_settings: dict, run_cfg: dict):
    papers = paper_markdowns(domain_cfg["markdown_dir"])
    if not llm_dicts:
        raise ValueError("no selected extraction models")
    for paper, path in papers.items():
        text = turnIntoPureText(path)
        for llm_key, llm in llm_dicts.items():
            for q_idx, question in enumerate(prompts.questions_for_domain(domain), start=1):
                provider = llm_settings.get(llm.get("source"), {})
                metadata = dict(paper_index=paper, question_index=q_idx, round_index=run_cfg["round_index"],
                                model=model_name(llm), paper_fingerprint=fingerprint(text))
                metadata["request_fingerprint"] = fingerprint({
                    "version": 2, "paper": metadata["paper_fingerprint"], "model": llm['model'], "source": llm.get('source'),
                    "endpoint": provider.get("url"), "json_mode": provider.get("supports_json_mode", True),
                    "domain": domain_cfg["domain_knowledge"], "system": prompts.message_system_v2,
                    "instruction": prompts.message_system_v2_output, "question": question,
                    "temperature": run_cfg["temperature"], "round": run_cfg["round_index"],
                })
                output = save_dataframe(llm, paper, q_idx, domain_cfg["examiner_output"], run_cfg["round_index"])
                yield output, metadata, text, llm_key, llm, question


def expected_tasks(domain: str, domain_cfg: dict, llm_dicts: dict, llm_settings: dict, run_cfg: dict) -> dict:
    if not llm_dicts:
        return {}
    return {output: metadata for output, metadata, *_ in domain_tasks(domain, domain_cfg, llm_dicts, llm_settings, run_cfg)}


# ---- Stage 1: Examiner — Main loop for a domain ----
# For each paper × each LLM × each question:
#   1. Skip if output CSV already exists (resume-safe)
#   2. Call run_llm_prompt_mode() with the truncated markdown
#   3. Parse JSON response → DataFrame
#   4. On request or parse failure: save raw text to _invalid.txt and leave the task resumable
#   5. Sleep for rate limiting (59s for low-limit, 1s for high-limit)
#   6. Save CSV with metadata columns (paper_index, model, tokens, time, question_index)
def run_examiner_for_domain(domain: str, domain_cfg: dict, llm_dicts: dict, llm_settings: dict, run_cfg: dict) -> None:
    failed = 0
    for output, metadata, text, llm_key, llm, question in domain_tasks(domain, domain_cfg, llm_dicts, llm_settings, run_cfg):
        paper, q_idx = metadata["paper_index"], metadata["question_index"]
        if Path(output).exists() and _successful_output(output, metadata, domain):
            print(f"Paper {paper} | {llm_key} - Question {q_idx} ... exists")
            continue
        # Sidecars retain canonical identities; filenames are display tokens only.
        save_json(Path(output).with_suffix(".meta.json"), metadata)
        start, raw = time.time(), None
        try:
            raw, token = run_llm_prompt_mode(llm, llm_settings, text, paper, q_idx, question,
                domain_cfg["domain_knowledge"], round_index=run_cfg["round_index"], temperature=run_cfg["temperature"])
            result = _df_from_result(raw)
            validate_answers(result, domain, q_idx)
        except Exception as exc:
            failed += 1
            detail = error_details(exc, llm_settings)
            write_invalid_text(output, f"ERROR {detail}\nRAW: {raw}" if raw is None else raw)
            print(f"Paper {paper} | {llm_key} - Question {q_idx} ... failed: {detail}")
            continue
        for key, value in metadata.items():
            result[key] = value
        result["total_tokens"] = token
        result["time_consumption"] = time.time() - start
        emission = prompts.emission_type_for_question(domain, q_idx)
        if emission:
            result["emission_type"] = emission
        write_dataframe(result, output)
        invalid_path(output).unlink(missing_ok=True)
        sleep_for_rate_limit(llm)
        print(f"Paper {paper} | {llm_key} - Question {q_idx} ... finished")
    if failed:
        raise RuntimeError(f"examiner failed: {failed} tasks; inspect *_invalid.txt and rerun")
