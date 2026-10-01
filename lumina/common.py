# -*- coding: utf-8 -*-
"""Shared utilities for the LUMINA aqua/wildfire pipeline."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from contextlib import contextmanager
from numbers import Real
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
  | references(?:\s+and\s+notes)?
  | bibliography
  | acknowledg(?:e)?ments
  | appendi(?:x|ces)
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
    with atomic_output(file_path) as temp:
        temp.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        logging.info("Content saved to %s", file_path)


def load_json(file_path: str | Path) -> Optional[Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        logging.error("Failed to load JSON from %s: %s", file_path, e)
        return None


def refineJsonString(str_content: str) -> Dict[str, Any]:
    if not isinstance(str_content, str) or not str_content.strip():
        raise ValueError("empty/null model content")
    json_scope = [str_content.index("{"), str_content.rfind("}") + 1]
    json_to_evaluate = str_content[json_scope[0] : json_scope[-1]]
    open_braces = len(re.findall(r"\{", json_to_evaluate))
    close_braces = len(re.findall(r"\}", json_to_evaluate))
    if close_braces < open_braces:
        json_to_evaluate += "\n}"
    parsed = json5.loads(json_to_evaluate)
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


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
    matches = list(re.finditer(REF_SECTION_PATTERN, text, re.IGNORECASE | re.MULTILINE | re.VERBOSE))
    body_heading = r"^\s*(?:\#{1,6}\s*)?(?:abstract|introduction|摘要|引言)\s*$"
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if re.search(body_heading, text[match.end():end], re.IGNORECASE | re.MULTILINE):
            continue  # A contents entry preceding the actual article body.
        return text[:match.start()]
    return text


def paper_prefix_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    head = re.split(r"[_-]", stem, maxsplit=1)[0]
    return head.zfill(2) if head.isdigit() else stem


def save_dataframe(llm: Dict[str, Any], paper_index: str | int, q_index: int, output_path: str | Path, round_index: int = 1) -> str:
    model_name = llm["model"].split("/")[-1]
    temp_dir = Path(output_path) / f"Paper_{str(paper_index).zfill(2)}"
    ensure_directory_exists(temp_dir)
    return str(
        temp_dir
        / f"{filename_token(model_name.lower())}_P{str(paper_index).zfill(2)}_Q{str(q_index).zfill(2)}_R{str(round_index).zfill(2)}.csv"
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


def write_invalid_text(output_file: str | Path, raw: object) -> None:
    with atomic_output(invalid_path(output_file)) as temp:
        temp.write_text("EMPTY_RESPONSE: null content" if raw is None else str(raw), encoding="utf-8")


def invalid_path(output_file: str | Path) -> Path:
    path = Path(output_file)
    return path.with_name(path.stem + "_invalid.txt")


@contextmanager
def atomic_output(file_path: str | Path):
    """Publish a complete file; an interrupted write leaves the previous file intact."""
    path = Path(file_path)
    ensure_directory_exists(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as handle:
        temp = Path(handle.name)
    try:
        yield temp
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_dataframe(frame: pd.DataFrame, file_path: str | Path) -> None:
    with atomic_output(file_path) as temp:
        if temp.suffix == ".xlsx":
            frame.to_excel(temp, index=False)
        else:
            frame.to_csv(temp, sep="\t", index=False)


@contextmanager
def pipeline_lock(path: str | Path):
    path = Path(path)
    ensure_directory_exists(path.parent)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another run or interrupted-run lock exists: {path}; verify the recorded PID before removing it") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def fingerprint(content: object) -> str:
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def error_details(exc: BaseException, settings: dict) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for provider in settings.values():
        if isinstance(provider, dict) and provider.get('key'):
            message = message.replace(str(provider['key']), '[REDACTED]')
    return message


def filename_token(text: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).strip(" .")[:100]
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    return safe if safe and safe == text and safe.lower() not in reserved else (safe or "model") + "-" + fingerprint(text)[:12]


def canonical_paper_id(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError("missing paper_index")
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not text:
        raise ValueError("empty paper_index")
    return text.zfill(2) if text.isdigit() else text


def paper_markdowns(markdown_dir: str | Path) -> dict[str, Path]:
    papers: dict[str, Path] = {}
    for path in sorted(Path(markdown_dir).glob("*.md")):
        paper = paper_prefix_from_path(path)
        if paper in papers:
            raise ValueError(f"duplicate paper ID collision {paper}: {papers[paper].name}, {path.name}")
        if not turnIntoPureText(path).strip():
            raise ValueError(f"empty paper body: {path}")
        papers[paper] = path
    if not papers:
        raise ValueError(f"no Markdown inputs: {markdown_dir}")
    return papers


def candidate_id(row) -> str:
    """Stable identity across Excel round trips, unrelated to row order."""
    fields = ["paper_index", "question_index", "round_index", "model", "item", "value", "unit",
              "evidence", "confidence_lv", "mce", "experimental", "request_fingerprint", "paper_fingerprint"]
    content = {}
    for field in fields:
        value = row.get(field)
        if value is None or pd.isna(value):
            value = None
        elif field in {'value', 'unit', 'evidence', 'mce', 'experimental'} and isinstance(value, str) and value.strip().lower() in {'', 'none', 'nan', 'nat', 'null', 'na', 'n/a'}:
            value = None
        elif field == "paper_index":
            value = canonical_paper_id(value)
        elif field in {"question_index", "round_index"}:
            value = int(value)
        elif isinstance(value, Real) and not isinstance(value, bool):
            value = int(value) if math.isfinite(value) and value == int(value) else float(value)
        content[field] = value
    return fingerprint(content)


def read_composite(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(path, keep_default_na=False, dtype={"paper_index": str, "model": str})
    required = {"paper_index", "question_index", "round_index", "model", "item", "value",
                "evidence", "confidence_lv", "request_fingerprint", "paper_fingerprint", "candidate_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"legacy/incomplete composite {path}; rerun examiner and composite (missing {sorted(required - set(frame.columns))})")
    frame["paper_index"] = frame["paper_index"].map(canonical_paper_id)
    for field in ['question_index', 'round_index']:
        values = pd.to_numeric(frame[field], errors='raise')
        if values.isna().any() or (values < 1).any() or (values % 1 != 0).any():
            raise ValueError(f'{path}: {field} must be positive integers')
        frame[field] = values.astype(int)
    return frame


def validate_answers(frame: pd.DataFrame, domain: str | None = None, question_index: int | None = None) -> None:
    required = {"item", "value", "evidence", "confidence_lv"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"empty or invalid answer schema; required {sorted(required)}")
    if frame["item"].isna().any() or frame["item"].astype(str).str.strip().isin({"", "ilegal", "invalid_syntax"}).any():
        raise ValueError("invalid answer item")
    conf = pd.to_numeric(frame["confidence_lv"], errors="coerce")
    if conf.isna().any() or not conf.between(-1, 100).all():
        raise ValueError("confidence_lv must be numeric from -1 to 100")
    for field in ["value", "evidence"]:
        if frame[field].map(lambda v: isinstance(v, (dict, list))).any():
            raise ValueError(f"{field} must be a scalar or null")
    if frame['evidence'].map(lambda v: not (isinstance(v, str) or v is None or pd.isna(v))).any():
        raise ValueError('evidence must be text or null')
    if domain is None:
        return
    numeric = question_index != 1 and not (domain == 'aqua' and question_index == 2)
    if numeric and frame['value'].map(lambda v: isinstance(v, bool) or
        (isinstance(v, Real) and not pd.isna(v) and not math.isfinite(v))).any():
        raise ValueError('numeric answer value must be finite and not boolean')
    allowed = {"Study_location", "Study_period"}
    if domain == "aqua" and question_index == 1:
        allowed |= {"Study_location_detail", "Latitude", "Longitude"}
    if question_index == 1 or (domain == "aqua" and question_index == 2):
        if domain == "aqua" and question_index == 2:
            allowed = {"Specie"}
        if not frame["item"].isin(allowed).all():
            raise ValueError(f"unexpected metadata item; allowed {sorted(allowed)}")
    elif domain == "aqua":
        if "unit" not in frame.columns:
            raise ValueError("Aqua numeric answers require a unit field (null permitted only for missing values)")
        missing_strings = {'', '999999', 'none', 'nan', 'null', 'n/a', 'na', 'not provided', 'not specified', '[]'}
        valid_value = frame["value"].notna() & ~frame["value"].astype(str).str.strip().str.lower().isin(missing_strings)
        missing_unit = frame["unit"].isna() | frame["unit"].astype(str).str.strip().str.lower().isin(missing_strings)
        if (valid_value & ~frame['unit'].map(lambda v: isinstance(v, str))).any():
            raise ValueError('Aqua unit must be text')
        if (valid_value & missing_unit).any():
            raise ValueError("Aqua numeric value has no unit")


def model_name(llm: Dict[str, Any]) -> str:
    return llm["model"].split("/")[-1]


def sleep_for_rate_limit(llm: Dict[str, Any]) -> None:
    import time

    time.sleep(59 if llm.get("limit_token", 1000000) <= 50000 else 1)
