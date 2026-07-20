# -*- coding: utf-8 -*-
"""Shared utilities for the LUMINA aqua/wildfire pipeline."""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import json5
import pandas as pd
import tiktoken


REF_SECTION_PATTERN = r"""^
\s*
(?:\#{1,6}\s*)?               # optional markdown heading marks
(?:[*_]{0,2})                 # optional bold/italic marks
(?:[■●\-–—]*)\s*              # optional bullets
(?:\[)?                       # optional left bracket
(?:                           # keywords
    (?:参\s*考\s*文\s*献)
  | (?:致\s*谢)
  | (?:附\s*录)
  | references
  | bibliography
  | acknowledg(?:e)?ments
)
(?:\])?                       # optional right bracket
(?:                           # optional parenthesized English
    [\(\（] \s*
    (?:references|bibliography|acknowledg(?:e)?ments)
    \s* [\)\）]
)?
(?:[*_]{0,2})
[\s:：]*
$
"""


def clean_excel_str(s: Any) -> Any:
    if isinstance(s, str):
        s = re.sub(r"[\x00-\x1F\x7F]", "", s)
        s = re.sub(r"<.*?>", "", s)
    return s


def log_time(func):
    def wrapper(*args, **kwargs):
        now = datetime.datetime.now()
        print(f"Function Starts at: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        return func(*args, **kwargs)

    return wrapper


def ensure_directory_exists(directory: str | Path) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(file_path: str | Path, content: Any, verbose: int = 0) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=4)
        if verbose:
            logging.info("Content saved to %s", file_path)
    except Exception as e:  # pragma: no cover
        logging.error("Failed to save JSON to %s: %s", file_path, e)


def load_json(file_path: str | Path) -> Optional[Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        logging.error("Failed to load JSON from %s: %s", file_path, e)
        return None


def refineJsonString(str_content: str) -> Dict[str, Any]:
    json_scope = [str_content.index("{"), str_content.rfind("}") + 1]
    json_to_evaluate = str_content[json_scope[0] : json_scope[-1]]
    open_braces = len(re.findall(r"\{", json_to_evaluate))
    close_braces = len(re.findall(r"\}", json_to_evaluate))
    if close_braces < open_braces:
        json_to_evaluate += "\n}"
    return json5.loads(json_to_evaluate)


def token_calculator(text: str, verbose: int = 0, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if verbose:
        print("Token 数量:", len(tokens))
    return len(tokens)


def turnIntoPureText(markdown_path: str | Path) -> str:
    """Return markdown content before references/acknowledgments/appendix.

    Note: the old template computed ``before_refs`` but returned ``text`` by mistake.
    This version returns the truncated text.
    """
    with open(markdown_path, encoding="utf-8") as f:
        text = f.read()
    match = re.search(REF_SECTION_PATTERN, text, re.IGNORECASE | re.MULTILINE | re.VERBOSE)
    return text[: match.start()] if match else text


def paper_prefix_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    head = re.split(r"[_-]", stem, maxsplit=1)[0]
    return head.zfill(2) if head.isdigit() else head


def save_dataframe(llm: Dict[str, Any], paper_index: str | int, q_index: int, output_path: str | Path, round_index: int = 1) -> str:
    model_name = llm["model"].split("/")[-1]
    temp_dir = Path(output_path) / f"Paper_{str(paper_index).zfill(2)}"
    ensure_directory_exists(temp_dir)
    return str(
        temp_dir
        / f"{model_name.lower()}_P{str(paper_index).zfill(2)}_Q{str(q_index).zfill(2)}_R{str(round_index).zfill(2)}.csv"
    )


def invalid_examiner_result() -> Dict[str, Any]:
    return {
        "invalid_syntax": {
            "value": None,
            "description": None,
            "confidence_lv": None,
        }
    }


def invalid_cross_result() -> Dict[str, Any]:
    return {"existing_flag": 999999, "direct_quote": None}


def error_cross_result() -> Dict[str, Any]:
    return {"existing_flag": -1, "direct_quote": "ERROR"}


def write_invalid_text(output_file: str | Path, raw: str) -> None:
    raw_path = str(output_file).replace(".csv", "_invalid.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)


def model_name(llm: Dict[str, Any]) -> str:
    return llm["model"].split("/")[-1]


def sleep_for_rate_limit(llm: Dict[str, Any]) -> None:
    import time

    time.sleep(59 if llm.get("limit_token", 1000000) <= 50000 else 1)
