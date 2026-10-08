"""One explicit, receipted HTTP attempt at a time; no hidden SDK retries."""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import requests
from openai import OpenAI

from ..common import error_details, fingerprint, save_json, turnIntoPureText
from .budget import AttemptLimit, Budget, BudgetExceeded, UnknownRequest
from .contracts import ResearchSpecification, TaskKey, validate_manifest, verify_inputs
from .store import Store


class ControlStop(RuntimeError):
    """Pause, budget or unresolved request. Core loops must propagate this."""


class RunContext:
    def __init__(self, run_dir: str | Path, config):
        self.run_dir = Path(run_dir).resolve()
        self.config = config
        self.store = Store(self.run_dir)
        try:
            manifest = validate_manifest(self.run_dir, self.store.snapshot()["run"]["spec_hash"])
            self.spec = manifest["specification"]
            verify_inputs(self.run_dir)
            request = {k: self.spec[k] for k in ("domain", "rounds", "budget", "pricing", "smoke_size", "allow_pdf_resources")}
            request["papers"] = [dict(path=str(self.run_dir / "inputs" / (p["paper_uid"] + p["suffix"])),
                                      paper_uid=p["paper_uid"]) for p in self.spec["papers"]]
            request["advisor"] = ({k: self.spec["advisor"][k] for k in ("model", "source")}
                                  if self.spec["advisor"] else None)
            request["smoke_papers"] = self.spec["smoke_paper_uids"]
            current = ResearchSpecification.from_config(request, config)
            if current.fingerprint != manifest["specification_hash"]:
                raise ValueError("current config or scientific code differs from the frozen specification; use the original revision or create a new run")
            self.budget = Budget(self.store, self.spec)
        except BaseException:
            self.store.close()
            raise
        self.current: TaskKey | None = None
        self._last_response_file: Path | None = None
        self.heartbeat = None

    def close(self):
        self.store.close()

    def key(self, stage: str, paper_uid="", question=None, **fields) -> TaskKey:
        return TaskKey(run_id=self.run_dir.name, stage=stage, paper_uid=paper_uid, question=question, **fields)

    def check(self, *, diagnostic=False) -> None:
        if self.heartbeat is not None:
            self.heartbeat()
        state = self.store.snapshot()["run"]["state"]
        if self.store.paused_requested():
            if state not in {"PAUSED", "DONE", "FATAL_ERROR"}:
                self.store.set_state("PAUSED", "pause requested; no new HTTP dispatched")
            raise ControlStop("run paused")
        diagnostic = diagnostic and self.spec["advisor"] is not None
        if state in {"PAUSED", "HUMAN_GATE_SMOKE", "FATAL_ERROR", "DONE"} or state == "HUMAN_GATE" and not diagnostic:
            raise ControlStop(f"run is {state}; explicit resolution is required")
        totals = self.budget.totals()
        limits = self.spec["budget"]
        if self.budget.over_budget():
            self._gate("budget", self.run_dir.name, "a durable budget stop forbids automatic resume")
        if totals["known_tokens"] + totals["held_tokens"] > limits["max_tokens"] or totals["known_cost"] + totals["held_cost"] > math.nextafter(limits["max_cost"], math.inf) or totals["calls"] > limits["max_calls"]:
            self._gate("budget", self.run_dir.name, "recorded usage exceeds frozen budget; automatic resume refused")
        if totals["runtime"] > limits["max_runtime"]:
            self._gate("budget", self.run_dir.name, "max_runtime reached")

    def _gate(self, kind: str, target: str, reason: str):
        reason = self.redact(reason)
        existing = [g for g in self.store.gates() if g["kind"] == kind and g["target"] == target and g["status"] == "pending"]
        gate_id = existing[0]["gate_id"] if existing else self.store.create_gate(kind, target, reason)
        state = self.store.snapshot()["run"]["state"]
        if state not in {"HUMAN_GATE", "FATAL_ERROR", "DONE"}:
            self.store.set_state("HUMAN_GATE", reason, task_id=self.current.task_id if self.current else None)
        self.store.export_views()
        raise ControlStop(f"{kind}: {reason}; gate={gate_id}")

    @contextmanager
    def task_scope(self, key: TaskKey):
        if key.run_id != self.run_dir.name:
            raise ValueError("task belongs to another run")
        if key.paper_uid and key.paper_uid not in {p["paper_uid"] for p in self.spec["papers"]}:
            raise ValueError("task paper is outside the frozen specification")
        if key.stage != "prepare" and key.round_index not in self.spec["rounds"]:
            raise ValueError("task round is outside the frozen specification")
        if key.question is not None and key.question not in {q["index"] for q in self.spec["questions"]}:
            raise ValueError("task question is outside the frozen specification")
        selected = {m["model"] for m in self.spec["models"]}
        if key.paper_uid and key.stage in {"examiner", "cross", "evidence_embedding"} and key.source_model not in selected:
            raise ValueError("task source model is outside the frozen specification")
        if key.paper_uid and key.stage == "cross" and (key.verifier_model not in selected or key.source_model == key.verifier_model):
            raise ValueError("task verifier must be an independent frozen model")
        self.check(diagnostic=key.stage == "advisor" and self.spec["advisor"] is not None and
                   key.source_model == self.spec["advisor"]["model"])
        previous = self.current
        self.current = key
        self.store.put_task(key.task_id, key.stage, key.to_dict())
        try:
            yield
        except BaseException as exc:
            status = "unresolved" if isinstance(exc, (ControlStop, KeyboardInterrupt)) else "failed"
            detail = self.redact(error_details(exc, self.config.LLM_SETTINGS))
            if isinstance(exc, (ControlStop, KeyboardInterrupt)) and self.store.valid_task(key.task_id):
                self.store.event("task_control_stop", {"task_id": key.task_id, "reason": detail,
                                                        "retained_status": "succeeded"})
            else:
                self.store.finish_task(key.task_id, status, [], error=detail)
            raise
        finally:
            self.current = previous

    def valid(self, key: TaskKey) -> bool:
        return self.store.valid_task(key.task_id)

    def completed(self, key: TaskKey, artifacts: list, cached=False):
        self.store.put_task(key.task_id, key.stage, key.to_dict())
        self.store.finish_task(key.task_id, "succeeded", artifacts)
        if cached:
            self.store.event("cache_hit", {"task_id": key.task_id, "stage": key.stage})

    def failed(self, key: TaskKey, error: str, artifacts=()):
        self.store.put_task(key.task_id, key.stage, key.to_dict())
        self.store.finish_task(key.task_id, "failed", list(artifacts), error=self.redact(error))

    @staticmethod
    def reraise_control(exc: BaseException):
        if isinstance(exc, (ControlStop, BudgetExceeded, UnknownRequest, AttemptLimit, KeyboardInterrupt)):
            raise exc

    def sleep(self, seconds: float):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check(diagnostic=self.current is not None and self.current.stage == "advisor")
            time.sleep(min(.2, max(0., deadline - time.monotonic())))

    def redact(self, value):
        if isinstance(value, str):
            for provider in self.config.LLM_SETTINGS.values():
                if provider.get("key"):
                    value = value.replace(str(provider["key"]), "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {k: ("[REDACTED]" if k.lower() in {"api_key", "authorization", "access_token"} else self.redact(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        return value

    def _allowed(self, model: dict, settings: dict, *, embedding=False):
        candidates = [self.spec["embedding"]] if embedding else self.spec["models"] + ([self.spec["advisor"]] if self.spec["advisor"] else [])
        match = next((m for m in candidates if m["model"] == model["model"] and m["source"] == model["source"]), None)
        provider = settings.get(model.get("source"), {})
        url = (model.get("url") if embedding else None) or provider.get("url")
        if match is None or url != match["endpoint"] or (not embedding and provider.get("supports_json_mode", True) != match["supports_json_mode"]):
            raise ValueError("request model/provider differs from frozen specification")
        if not isinstance(provider.get("key"), str) or not provider["key"].strip():
            raise ValueError("provider credentials missing; no request dispatched")
        return match

    def _timeout(self, model=None):
        return max(.01, min((model or {}).get('timeout_seconds', 120.),
                           self.spec["budget"]["max_runtime"] - self.budget.totals()["runtime"]))

    @staticmethod
    def _usage(response: dict, *, embedding=False):
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = 0 if embedding else usage.get("completion_tokens", usage.get("output_tokens"))
        if type(input_tokens) is not int or type(output_tokens) is not int or min(input_tokens, output_tokens) < 0:
            return None
        return dict(input_tokens=input_tokens, output_tokens=output_tokens)

    @staticmethod
    def _confirmed_rejection(exc):
        # A generic timeout/5xx/connection error is NOT proof of non-execution.
        response = getattr(exc, "response", None)
        if response is None:
            return None
        try:
            error = response.json().get("error", {})
            code = error.get("code") or error.get("type")
            status = getattr(response, "status_code", None)
            if status == 429 and code in {"rate_limit_exceeded", "rate_limit"}:
                return "retry"
            permanent = {400: {"context_length_exceeded", "invalid_request_error", "invalid_request"},
                         401: {"invalid_api_key", "authentication_error"},
                         403: {"permission_denied", "insufficient_permissions", "access_denied"},
                         404: {"model_not_found", "not_found"}}
            if code in permanent.get(status, set()):
                return "stop"
        except (ValueError, AttributeError, TypeError):
            pass
        return None

    def _request(self, model: dict, kind: str, payload: dict, invoke) -> dict:
        if self.current is None:
            raise RuntimeError("paid requests require a task scope")
        self.check(diagnostic=kind == "advisor")
        key = self.current
        request_hash = fingerprint(dict(payload=payload, kind=kind, spec=self.spec["pricing"][model["model"]],
                                        endpoint=model["endpoint"], source=model["source"]))
        prior = self.store._conn.execute("SELECT request_hash FROM attempts WHERE task_id=?", (key.task_id,)).fetchall()
        if any(row["request_hash"] != request_hash for row in prior):
            self._gate("request_changed", key.task_id, "logical task request changed inside a frozen run")
        try:
            cached = self.budget.cached(key.task_id, request_hash)
            source_task_id = key.task_id
            if cached is None and not prior and kind == "embedding":
                shared = self.store._conn.execute(
                    "SELECT task_id FROM attempts WHERE request_hash=? AND kind='embedding' "
                    "AND status IN ('received','imported') ORDER BY created_at LIMIT 1", (request_hash,)).fetchone()
                if shared:
                    source_task_id = shared["task_id"]
                    cached = self.budget.cached(source_task_id, request_hash)
        except UnknownRequest as exc:
            self._gate("receipt_integrity", key.task_id, str(exc))
        if cached:
            response = cached["response"]
            attempt_id = cached["attempt_id"]
            self.store.event("response_reused", {"task_id": key.task_id, "source_task_id": source_task_id,
                                                  "attempt_id": attempt_id, "kind": kind})
        else:
            price = self.spec["pricing"][model["model"]]
            if kind == 'embedding' and model.get('rate_limit_seconds'):
                self.sleep(model['rate_limit_seconds'])
            estimated_input = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 128
            if estimated_input > price["max_input_tokens"]:
                self._gate("input_bound", key.task_id, "request text exceeds approved conservative input bound")
            while True:
                self.check(diagnostic=kind == "advisor")
                state = self.store.snapshot()["run"]["state"]
                if kind != "advisor":
                    if state not in {"SMOKE", "BATCH"}:
                        self._gate("phase_violation", key.task_id, "scientific HTTP is only allowed in smoke or approved batch")
                    if state == "SMOKE" and key.paper_uid not in self.spec["smoke_paper_uids"]:
                        self._gate("phase_violation", key.task_id, "paper is outside the explicitly selected smoke scope")
                    if state == "BATCH" and not any(g["kind"] == "smoke" and g["status"] == "approved" and g["decision"].get("scope") == "batch" for g in self.store.gates()):
                        self._gate("phase_violation", key.task_id, "batch has no recorded human approval")
                try:
                    attempt_id = self.budget.reserve(key.task_id, request_hash, kind, model["model"],
                                                     price["max_input_tokens"], 0 if kind == "embedding" else price["max_output_tokens"])
                    self.budget.dispatch(attempt_id)
                except (BudgetExceeded, AttemptLimit) as exc:
                    self._gate("budget", key.task_id, str(exc))
                except UnknownRequest as exc:
                    unresolved = [r for r in self.store._conn.execute(
                        "SELECT attempt_id FROM attempts WHERE task_id=? AND request_hash=? AND status IN ('dispatched','unknown')",
                        (key.task_id, request_hash))]
                    self._gate("unknown_request", unresolved[0][0] if unresolved else key.task_id, str(exc))
                try:
                    response = self.redact(invoke())
                except BaseException as exc:
                    detail = self.redact(error_details(exc, self.config.LLM_SETTINGS))
                    rejected = self._confirmed_rejection(exc)
                    try:
                        self.budget.fail(attempt_id, detail, confirmed_not_executed=bool(rejected))
                    except UnknownRequest:
                        pass  # The unknown receipt is durable; the gate below is mandatory.
                    if rejected == "retry":
                        self.store.event("request_rejected", {"attempt_id": attempt_id, "error": detail})
                        self.sleep(1.)
                        continue
                    if rejected == "stop":
                        self._gate("confirmed_rejection", attempt_id, detail + "; request was rejected before execution")
                    self._gate("unknown_request", attempt_id, detail)
                try:
                    self.budget.receive(attempt_id, response, self._usage(response, embedding=kind == "embedding"))
                except BudgetExceeded as exc:
                    self._gate("budget", attempt_id, str(exc))
                break
        raw_file = self.run_dir / "checkpoints" / "responses" / f"{attempt_id}.json"
        save_json(raw_file, dict(attempt_id=attempt_id, task_id=source_task_id if cached else key.task_id,
                                 request_hash=request_hash, response=response))
        self._last_response_file = raw_file
        return response

    def chat(self, llm: dict, settings: dict, messages: list[dict], temperature=.01):
        if self.current is None:
            raise RuntimeError("paid requests require a task scope")
        expected = {"examiner": self.current.source_model, "cross": self.current.verifier_model,
                    "advisor": self.spec["advisor"]["model"] if self.spec["advisor"] else None}
        if self.current.stage not in expected or expected[self.current.stage] != llm["model"]:
            raise ValueError("chat model does not match its frozen task role")
        if self.current.stage == "cross" and self.current.source_model == llm["model"]:
            raise ValueError("self-verification is forbidden")
        if temperature != self.spec["run"]["temperature"]:
            raise ValueError("chat temperature differs from frozen specification")
        if self.current.stage == "examiner":
            self._validate_extraction_messages(messages)
        match = self._allowed(llm, settings)
        provider = settings[llm["source"]]
        price = self.spec["pricing"][llm["model"]]
        payload = dict(model=llm["model"], messages=messages, temperature=temperature, stream=False,
                       max_tokens=price["max_output_tokens"])
        if match["supports_json_mode"]:
            payload["response_format"] = {"type": "json_object"}

        def invoke():
            client = OpenAI(api_key=provider["key"], base_url=provider["url"], max_retries=0, timeout=self._timeout(match))
            try:
                return client.chat.completions.create(**payload).model_dump(mode="json")
            finally:
                client.close()

        response = self._request(match, "advisor" if self.current and self.current.stage == "advisor" else "chat", payload, invoke)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("chat response has no choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish == "length":
            raise ValueError("truncated chat response; raw receipt retained, no automatic requery")
        if finish != "stop":
            raise ValueError(f"chat finish reason is not complete: {finish}")
        message = choice.get("message", {})
        content = message.get("content")
        if message.get("refusal") or not isinstance(content, str) or not content.strip():
            raise ValueError("chat response empty or refused")
        usage = self._usage(response)
        return content, sum(usage.values()) if usage is not None else None

    def _validate_extraction_messages(self, messages):
        key = self.current
        question = next((q for q in self.spec["questions"] if q["index"] == key.question), None)
        if question is None or not isinstance(messages, list) or len(messages) != 3:
            raise ValueError("extraction task needs its frozen question and three messages")
        definition = self.spec['question_set']
        prompt = definition['questions'][key.question - 1]['prompt'].strip("\n")
        if fingerprint(prompt) != question["prompt_hash"]:
            raise ValueError("question prompt differs from the frozen specification")
        prepared = self.run_dir / "outputs" / "prepared" / (key.paper_uid + ".md")
        if prepared.exists():
            if not self.valid(self.key("prepare", paper_uid=key.paper_uid)):
                raise ValueError("prepared paper has no valid receipt")
            path = prepared
        else:
            paper = next(p for p in self.spec["papers"] if p["paper_uid"] == key.paper_uid)
            if paper["suffix"] != ".md":
                raise ValueError("PDF must be prepared before extraction")
            path = self.run_dir / "inputs" / (key.paper_uid + ".md")
        templates = definition['templates']
        expected = [dict(role="system", content=templates['system'].format(domain=self.spec["domain_knowledge"])),
                    dict(role="user", content=templates['instruction'].format(content=turnIntoPureText(path))),
                    dict(role="user", content=prompt)]
        if messages != expected:
            raise ValueError("extraction messages do not match the frozen paper and question")

    def embedding(self, text: str, model: dict, settings: dict):
        if self.current is None:
            raise RuntimeError("paid embeddings require a task scope")
        if self.current.stage not in {"embeddings", "evidence_embedding"} or not self.current.paper_uid:
            raise ValueError("embedding requires a frozen paper task")
        match = self._allowed(model, settings, embedding=True)
        key = replace(self.current, chunk=fingerprint(text))
        base_key = self.current
        with self.task_scope(key):
            payload = dict(model=model["model"], input=str(text), encoding_format="float")

            def invoke():
                response = requests.post(match["endpoint"], json=payload,
                                         headers={"Authorization": "Bearer " + settings[model["source"]]["key"],
                                                  "Content-Type": "application/json"}, timeout=self._timeout(match))
                response.raise_for_status()
                try:
                    parsed = response.json()
                except ValueError:
                    return {"raw_response": response.text, "parse_error": "embedding response is not JSON"}
                return parsed if isinstance(parsed, dict) else {"raw_response": parsed, "parse_error": "embedding response is not an object"}

            response = self._request(match, "embedding", payload, invoke)
            data = response.get("data")
            if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
                raise ValueError("embedding response requires exactly one data item")
            vector = data[0].get("embedding")
            if not isinstance(vector, list) or not vector or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector) or not any(vector):
                raise ValueError("embedding must be a finite, nonzero numeric vector")
            self.completed(key, [self._last_response_file])
            if base_key.stage == "evidence_embedding":
                self.completed(base_key, [self._last_response_file])
            return vector
