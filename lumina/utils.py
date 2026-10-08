from __future__ import annotations

import time
from pathlib import Path


def model_name(llm: dict) -> str:
    return llm["model"].split("/")[-1]


def sleep_for_rate_limit(llm: dict, low_limit_seconds: float = 59.0, high_limit_seconds: float = 1.0,
                         runtime=None) -> None:
    limit_token = llm.get("limit_token", 10**9)
    seconds = llm.get('rate_limit_seconds', low_limit_seconds if limit_token <= 50000 else high_limit_seconds)
    if runtime is not None:
        runtime.sleep(seconds)
    else:
        time.sleep(seconds)


def get_ext_files(directory: str | Path, ext: str) -> list[str]:
    return sorted(str(p) for p in Path(directory).glob(f"*.{ext}"))
