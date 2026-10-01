"""Explicit, provenance-checked reuse. No automatic shared result directory."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from .budget import Budget
from .contracts import TaskKey, file_hash, validate_manifest, verify_inputs
from .lease import RunLease
from .store import Store

IMPORT_STAGES = {"prepare", "examiner", "embeddings", "evidence_embedding", "cross"}


def import_task(source_run, target_run, task_id: str, *, reason: str):
    source_run, target_run = Path(source_run).resolve(), Path(target_run).resolve()
    if source_run == target_run or not isinstance(reason, str) or not reason.strip():
        raise ValueError("import requires two different runs and an explicit reason")
    with ExitStack() as stack:
        for directory in sorted([source_run, target_run], key=str):
            stack.enter_context(RunLease(directory))
        source = stack.enter_context(Store(source_run))
        target = stack.enter_context(Store(target_run))
        manifests = [validate_manifest(run, store.snapshot()["run"]["spec_hash"])
                     for run, store in [(source_run, source), (target_run, target)]]
        for run in (source_run, target_run):
            verify_inputs(run)
        if manifests[0]["scientific_hash"] != manifests[1]["scientific_hash"] or manifests[0]["specification"]["pricing"] != manifests[1]["specification"]["pricing"]:
            raise ValueError("import signatures differ: inputs, models/endpoints, prompts, code, rounds, thresholds or request bounds")
        if target.snapshot()["run"]["state"] != "INPUT_READY":
            raise ValueError("imports are accepted only before target execution")
        original = source.task(task_id)
        if original is None or original["stage"] not in IMPORT_STAGES or not source.valid_task(task_id):
            raise ValueError("source task lacks valid, complete importable artifacts")
        key = TaskKey(**original["payload"])
        tasks = [original]
        if key.stage in {"embeddings", "evidence_embedding"} and not key.chunk:
            # The overall matrix depends on every chunk receipt; import the verified descendants too.
            tasks.extend(t for t in source.tasks(key.stage) if t["task_id"] != task_id and
                         t["payload"].get("chunk") and
                         replace(TaskKey(**t["payload"]), chunk="") == key)
        plans = []
        for task in tasks:
            if not source.valid_task(task["task_id"]):
                raise ValueError("source dependency has missing or changed artifacts")
            target_key = replace(TaskKey(**task["payload"]), run_id=target_run.name)
            existing = target.task(target_key.task_id)
            if existing and existing["status"] != "pending":
                raise ValueError("target task already has execution history; import will not overwrite it")
            if target._conn.execute("SELECT 1 FROM attempts WHERE task_id=?", (target_key.task_id,)).fetchone():
                raise ValueError("target already attempted this task; import would hide accounting")
            receipts = source._conn.execute("SELECT * FROM attempts WHERE task_id=?", (task["task_id"],)).fetchall()
            for receipt in receipts:
                if receipt["status"] not in {"received", "imported", "rejected"}:
                    raise ValueError("unresolved source request cannot be imported")
                if receipt["status"] in {"received", "imported"}:
                    Budget(source, manifests[0]["specification"]).cached(task["task_id"], receipt["request_hash"])
                    payload = json.loads(receipt["receipt"])
                    if payload.get("stop_reason"):
                        raise ValueError("source request exceeded its approved bounds")
            for artifact in task["artifacts"]:
                path = (target_run / artifact["path"]).resolve()
                if not path.is_relative_to(target_run):
                    raise ValueError("import artifact path escapes target run")
                if path.exists() and (not path.is_file() or file_hash(path) != artifact["sha256"]):
                    raise ValueError("import would overwrite a different target artifact")
            plans.append((task, target_key, receipts))
        imported = []
        for task, target_key, receipts in plans:
            artifacts = []
            for artifact in task["artifacts"]:
                destination = target_run / artifact["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_run / artifact["path"], destination)
                if file_hash(destination) != artifact["sha256"]:
                    raise ValueError("source artifact changed during import")
                artifacts.append(destination)
            target.put_task(target_key.task_id, target_key.stage, target_key.to_dict())
            with target.transaction() as conn:
                for receipt in receipts:
                    if receipt["status"] == "rejected":
                        continue  # Historical rejected attempts are provenance, not target retries.
                    payload = json.loads(receipt["receipt"])
                    payload.pop("dispatched_at", None)
                    payload["original_usage"] = payload.get("usage")
                    payload["imported_from"] = dict(run_id=source_run.name, task_id=task["task_id"],
                                                     attempt_id=receipt["attempt_id"], reason=reason,
                                                     specification_hash=manifests[0]["specification_hash"])
                    now = time.time()
                    conn.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (uuid.uuid4().hex, target_key.task_id, receipt["request_hash"], receipt["kind"],
                                  "imported", 0, 0., 0, 0., None, json.dumps(payload, ensure_ascii=False), now, now))
            target.finish_task(target_key.task_id, "succeeded", artifacts)
            target.event("task_imported", dict(source_run=source_run.name, source_task=task["task_id"],
                                                target_task=target_key.task_id, reason=reason,
                                                source_specification_hash=manifests[0]["specification_hash"],
                                                artifacts=task["artifacts"]))
            imported.append(target_key.task_id)
        target.export_views()
        return dict(status="imported", tasks=imported, source_run=source_run.name, target_run=target_run.name)
