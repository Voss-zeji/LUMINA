"""Freeze the inputs actually used by the scientific pipeline, without credentials."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .. import ensemble, prompts
from ..common import atomic_output, canonical_paper_id, fingerprint, save_json

FORMAT_VERSION = 1
SCIENTIFIC_FILES = (
    "prompts.py", "common.py", "llm.py", "utils.py", "preparation.py", "examiner.py", "composite.py",
    "cross_validation.py", "ensemble.py", "ensemble_utils_meta.py", "ensemble_utils_value.py",
)
RUN_FIELDS = {"round_index", "temperature", "chunk_size", "overlap_percent", "text_extension", "min_cross_scores"}
SPEC_FIELDS = {"domain", "papers", "rounds", "budget", "pricing", "smoke_size", "smoke_papers", "advisor", "allow_pdf_resources"}
BUDGET_FIELDS = {"max_calls", "max_tokens", "max_cost", "max_runtime"}
RESERVED_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _number(value, name: str, minimum=0, integer=False, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")
    return value


def endpoint(value) -> str:
    if not isinstance(value, str):
        raise ValueError("endpoint must be an explicit URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("endpoint must be an HTTP(S) URL without credentials, query or fragment")
    return value


def _model(model: dict, settings: dict, *, embedding=False) -> dict:
    source, name = model.get("source"), model.get("model")
    if not isinstance(name, str) or not name.strip() or not isinstance(source, str) or source not in settings:
        raise ValueError("model requires full model ID and an existing provider")
    provider = settings[source]
    result = dict(model=name, source=source,
                  endpoint=endpoint((model.get("url") if embedding else None) or provider.get("url")))
    if not embedding:
        json_mode = provider.get("supports_json_mode", True)
        if type(json_mode) is not bool:
            raise ValueError("supports_json_mode must be a boolean")
        result["supports_json_mode"] = json_mode
        result["limit_token"] = _number(model.get("limit_token", 10**9), "limit_token", strict=True, integer=True)
    return result


def _paper(raw) -> dict:
    if isinstance(raw, str):
        raw = {"path": raw}
    if not isinstance(raw, dict) or set(raw) - {"path", "paper_uid"} or not isinstance(raw.get("path"), str):
        raise ValueError("paper must be a path or {path, paper_uid}")
    path = Path(raw["path"]).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in {".md", ".pdf"} or path.stat().st_size == 0:
        raise ValueError(f"unsupported or empty input: {path.name}")
    if path.suffix.lower() == ".md" and not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"empty Markdown input: {path.name}")
    digest = file_hash(path)
    uid = raw.get("paper_uid", "p" + digest[:32])
    if not isinstance(uid, str) or not re.fullmatch(r"[A-Za-z0-9]{1,64}", uid):
        raise ValueError("paper_uid must contain 1–64 letters or digits")
    if uid.lower() in RESERVED_NAMES or canonical_paper_id(uid) != uid:
        raise ValueError("paper_uid must be portable and already canonical; prefer a letter-prefixed UID")
    return dict(paper_uid=uid, sha256=digest, suffix=path.suffix.lower(),
                display_name=path.name, source_path=str(path), size_bytes=path.stat().st_size)


def _identity(data: dict, *, science=False) -> dict:
    result = json.loads(json.dumps(data, allow_nan=False))
    result["papers"] = [dict(paper_uid=p["paper_uid"], sha256=p["sha256"], suffix=p["suffix"])
                        for p in result["papers"]]
    result["models"] = sorted([{k: v for k, v in m.items() if k != "key"} for m in result["models"]],
                              key=lambda m: (m["model"], m["source"], m["endpoint"]))
    if science:
        for key in ("budget", "pricing", "smoke_size", "smoke_paper_uids", "advisor", "allow_pdf_resources"):
            result.pop(key, None)
    return result


@dataclass(frozen=True)
class ResearchSpecification:
    """Canonical JSON makes nested scientific conditions immutable as well."""
    _json: str

    @property
    def data(self) -> dict:
        return json.loads(self._json)

    @property
    def fingerprint(self) -> str:
        return fingerprint(_identity(self.data))

    @property
    def scientific_hash(self) -> str:
        return fingerprint(_identity(self.data, science=True))

    def to_json(self) -> str:
        return self._json

    @classmethod
    def from_config(cls, request: dict, config) -> ResearchSpecification:
        if not isinstance(request, dict) or set(request) - SPEC_FIELDS:
            raise ValueError("unknown specification fields; credentials belong in the local config")
        domain = request.get("domain")
        if domain not in {"aqua", "wildfire"} or domain not in config.DOMAINS:
            raise ValueError("domain must be aqua or wildfire and configured")
        questions = prompts.questions_for_domain(domain)
        domain_cfg = config.DOMAINS[domain]
        if domain_cfg.get("questions") != list(range(1, len(questions) + 1)):
            raise ValueError("questions must match the scientific core; custom question selection is unsupported")
        if not isinstance(domain_cfg.get("domain_knowledge"), str) or not domain_cfg["domain_knowledge"].strip():
            raise ValueError("domain_knowledge must be explicit")
        raw_papers = request.get("papers")
        if not isinstance(raw_papers, list) or not raw_papers:
            raise ValueError("papers must be a nonempty explicit list")
        input_order = [_paper(raw) for raw in raw_papers]
        papers = sorted(input_order, key=lambda p: p["paper_uid"])
        for key in ("sha256", "paper_uid"):
            values = [p[key].casefold() for p in papers]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate paper {key}; resolve duplicate inputs explicitly before creating a run")
        keys = config.SELECTED_KEYS
        if not keys or len(set(keys)) != len(keys) or set(keys) - set(config.FULL_LLM_POOL):
            raise ValueError("selected model keys must be unique and configured")
        models = [dict(key=key, **_model(config.FULL_LLM_POOL[key], config.LLM_SETTINGS)) for key in keys]
        names = [m["model"].split("/")[-1].lower() for m in models]
        if len(set(names)) != len(names):
            raise ValueError("selected model names collide; independent model identities must be unique")
        embedding = _model(config.EMBEDDING_MODEL, config.LLM_SETTINGS, embedding=True)
        run = dict(config.RUN)
        if set(run) - {'min_consensus_models'} != RUN_FIELDS:
            raise ValueError("RUN must explicitly contain exactly the supported scientific settings")
        run['min_consensus_models'] = ensemble.consensus_threshold(names, run)
        _number(run["round_index"], "round_index", integer=True, strict=True)
        _number(run["temperature"], "temperature")
        if run["temperature"] > 2:
            raise ValueError("temperature must be at most 2")
        _number(run["chunk_size"], "chunk_size", integer=True, strict=True)
        _number(run["overlap_percent"], "overlap_percent")
        if run["overlap_percent"] >= 100:
            raise ValueError("overlap_percent must be less than 100")
        _number(run["text_extension"], "text_extension", integer=True)
        scores = run["min_cross_scores"]
        if not isinstance(scores, list) or any(type(s) is not int or not 1 <= s < len(models) for s in scores) or len(scores) != len(set(scores)):
            raise ValueError("min_cross_scores must be unique integers reachable by independent verifiers")
        rounds = request.get("rounds", [run["round_index"]])
        if not isinstance(rounds, list) or not rounds or any(type(r) is not int or r < 1 for r in rounds) or len(rounds) != len(set(rounds)):
            raise ValueError("rounds must be unique positive integers")
        rounds = sorted(rounds)
        # round_index describes legacy defaults; the explicit rounds list is authoritative here.
        run["round_index"] = rounds[0]
        budget = request.get("budget")
        if not isinstance(budget, dict) or set(budget) != BUDGET_FIELDS:
            raise ValueError("budget requires max_calls, max_tokens, max_cost and max_runtime")
        for key, value in budget.items():
            _number(value, key, integer=key in {"max_calls", "max_tokens"}, strict=True)
        advisor = request.get("advisor")
        if advisor is not None:
            if not isinstance(advisor, dict) or set(advisor) != {"model", "source"}:
                raise ValueError("advisor requires an explicitly chosen model and provider, or null")
            advisor = _model(advisor, config.LLM_SETTINGS)
        required_models = {m["model"] for m in models} | {embedding["model"]}
        if advisor:
            required_models.add(advisor["model"])
        pricing = request.get("pricing")
        if not isinstance(pricing, dict) or set(pricing) != required_models:
            raise ValueError("pricing must cover exactly every scientific, embedding and optional advisor model")
        price_fields = {"input_per_million", "output_per_million", "max_input_tokens", "max_output_tokens", "basis"}
        for model, price in pricing.items():
            if not isinstance(price, dict) or set(price) != price_fields or not isinstance(price["basis"], str) or not price["basis"].strip():
                raise ValueError(f"pricing for {model} requires rates, input/output bounds and price basis")
            for field in price_fields - {"basis"}:
                _number(price[field], field, integer=field.startswith("max_"), strict=field.startswith("max_"))
        smoke_size = _number(request.get("smoke_size", 1), "smoke_size", integer=True, strict=True)
        if smoke_size > len(papers):
            raise ValueError("smoke_size exceeds paper count")
        smoke_papers = request.get("smoke_papers")
        if smoke_papers is None:
            smoke_uids = [p["paper_uid"] for p in input_order[:smoke_size]]
        else:
            if not isinstance(smoke_papers, list) or not smoke_papers or any(not isinstance(p, str) for p in smoke_papers):
                raise ValueError("smoke_papers must explicitly name paper UIDs or input paths")
            by_uid = {p["paper_uid"]: p["paper_uid"] for p in papers}
            by_path = {p["source_path"]: p["paper_uid"] for p in papers}
            smoke_uids = [by_uid.get(p) or by_path.get(str(Path(p).resolve())) for p in smoke_papers]
            if any(p is None for p in smoke_uids) or len(set(smoke_uids)) != len(smoke_uids):
                raise ValueError("smoke_papers must be unique and belong to the frozen input list")
            if "smoke_size" in request and smoke_size != len(smoke_uids):
                raise ValueError("smoke_size conflicts with smoke_papers")
            smoke_size = len(smoke_uids)
        allow_pdf = request.get("allow_pdf_resources", False)
        if type(allow_pdf) is not bool:
            raise ValueError("allow_pdf_resources must be boolean")
        core = Path(__file__).resolve().parents[1]
        data = dict(format_version=FORMAT_VERSION, domain=domain, papers=papers,
                    domain_knowledge=domain_cfg["domain_knowledge"], models=models, embedding=embedding,
                    run=run, rounds=rounds, budget=budget, pricing=pricing, smoke_size=smoke_size,
                    smoke_paper_uids=sorted(smoke_uids),
                    advisor=advisor, allow_pdf_resources=allow_pdf,
                    questions=[dict(index=i, prompt_hash=fingerprint(q.strip("\n"))) for i, q in enumerate(questions, 1)],
                    scientific_code={name: file_hash(core / name) for name in SCIENTIFIC_FILES},
                    control_code={str(path.relative_to(core.parent)).replace("\\", "/"): file_hash(path)
                                  for path in [core.parent / "run_pipeline.py", *sorted((core / "agent").glob("*.py"))]},
                    policies=dict(cross="evidence existence and relevance", numeric="verified unique mode with distinct model consensus",
                                  consensus="Tvfy AND Tbsl; 00_full is diagnostic; ties are unaccepted",
                                  units="no automatic unit conversion", verification="all independent selected models"))
        serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        for provider in config.LLM_SETTINGS.values():
            credential = provider.get("key")
            if isinstance(credential, str) and credential and credential in serialized:
                raise ValueError("provider credential occurs in specification metadata; remove it before freezing")
        return cls(serialized)


@dataclass(frozen=True)
class TaskKey:
    run_id: str
    stage: str
    paper_uid: str = ""
    question: int | None = None
    source_model: str = ""
    verifier_model: str = ""
    round_index: int = 1
    candidate: str = ""
    chunk: str = ""

    def __post_init__(self):
        for field in ("run_id", "stage", "paper_uid", "source_model", "verifier_model", "candidate", "chunk"):
            if not isinstance(getattr(self, field), str):
                raise ValueError(f"task {field} must be a string")
        if not self.run_id or not self.stage or type(self.round_index) is not int or self.round_index < 1:
            raise ValueError("task requires run, stage and a positive integer round")
        if self.question is not None and (type(self.question) is not int or self.question < 1):
            raise ValueError("task question must be a positive integer, not a boolean or float")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def task_id(self) -> str:
        return fingerprint(self.to_dict())


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    expected: int
    completed: int
    failed: int = 0
    unresolved: int = 0
    artifacts: tuple = ()
    quality: tuple = ()


@dataclass(frozen=True)
class AttemptReceipt:
    attempt_id: str
    task_id: str
    status: str
    raw_path: str | None = None
    tokens: int | None = None
    cost: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class ActionProposal:
    action: str
    target: str
    reason: str


def create_run(spec: ResearchSpecification, root: str | Path, run_id: str | None = None) -> Path:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id) or run_id.lower() in RESERVED_NAMES:
        raise ValueError("run_id must be a safe directory name")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = root / run_id
    run.mkdir()  # Never overwrite or attach to a preexisting run implicitly.
    for directory in ("inputs", "outputs", "checkpoints", "reports"):
        (run / directory).mkdir()
    data = spec.data
    for paper in data["papers"]:
        target = run / "inputs" / (paper["paper_uid"] + paper["suffix"])
        with atomic_output(target) as staged:
            shutil.copyfile(paper["source_path"], staged)
            if file_hash(staged) != paper["sha256"]:
                raise ValueError("input changed after specification freeze; create a new specification")
    manifest = dict(format_version=FORMAT_VERSION, run_id=run_id,
                    created_at=datetime.now(timezone.utc).isoformat(), specification=data,
                    specification_hash=spec.fingerprint, scientific_hash=spec.scientific_hash)
    save_json(run / "manifest.json", manifest)
    return run


def validate_manifest(run_dir: str | Path, expected_hash: str | None = None) -> dict:
    run = Path(run_dir).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("run_id") != run.name:
        raise ValueError("manifest version or run identity mismatch")
    spec = ResearchSpecification(json.dumps(manifest["specification"], allow_nan=False))
    if spec.fingerprint != manifest.get("specification_hash") or spec.scientific_hash != manifest.get("scientific_hash"):
        raise ValueError("manifest fingerprint mismatch")
    if expected_hash is not None and spec.fingerprint != expected_hash:
        raise ValueError("manifest fingerprint differs from the authoritative ledger")
    return manifest


def verify_inputs(run_dir: str | Path) -> None:
    run = Path(run_dir).resolve()
    manifest = validate_manifest(run)
    for paper in manifest["specification"]["papers"]:
        file = (run / "inputs" / (paper["paper_uid"] + paper["suffix"])).resolve()
        if not file.is_relative_to(run / "inputs") or not file.is_file() or file_hash(file) != paper["sha256"]:
            raise ValueError(f"frozen input missing or changed: {paper['paper_uid']}")
