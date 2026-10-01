"""CLI worker supervision: the approved wall deadline also bounds blocked local work."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ..common import error_details, save_json
from .budget import Budget
from .contracts import validate_manifest
from .lease import RunLease
from .store import Store


def _expired(run):
    with RunLease(run), Store(run) as store:
        manifest = validate_manifest(run, store.snapshot()["run"]["spec_hash"])
        recovery = Budget(store, manifest["specification"]).recover_unfinished()
        state = store.snapshot()["run"]["state"]
        if state in {"DONE", "FATAL_ERROR"}:
            return {"state": state, "run_id": run.name}
        for attempt in recovery["marked_unknown"]:
            store.create_gate("unknown_request", attempt, "worker deadline expired after dispatch; execution outcome unknown")
        store.create_gate("budget", run.name, "approved wall runtime exhausted; worker terminated")
        if state != "HUMAN_GATE":
            store.set_state("HUMAN_GATE", "wall deadline exhausted; no further work admitted")
        store.event("worker_deadline", {"unresolved_attempts": recovery["marked_unknown"]})
        store.export_views()
        result = {"state": "HUMAN_GATE", "run_id": run.name, "reason": "max_runtime exhausted",
                  "budget": Budget(store, manifest["specification"]).totals(),
                  "scientific_validation": "NOT_EVALUATED"}
        save_json(run / "reports" / "deadline_report.json", result)
        return result


def supervise(run_dir, config_path, *, worker_command=None):
    run = Path(run_dir).resolve()
    with Store(run) as store:
        manifest = validate_manifest(run, store.snapshot()["run"]["spec_hash"])
        if store.snapshot()["run"]["state"] in {"DONE", "FATAL_ERROR"}:
            from .controller import execute
            from run_pipeline import load_config
            return execute(run, load_config(str(config_path)))
        remaining = manifest["specification"]["budget"]["max_runtime"] - Budget(store, manifest["specification"]).totals()["runtime"]
    if remaining <= 0:
        return _expired(run)
    command = worker_command or [sys.executable, "-m", "lumina.agent.worker", "--run-dir", str(run),
                                 "--config", str(Path(config_path).resolve())]
    try:
        child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=remaining,
                               cwd=Path(__file__).resolve().parents[2],
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for this child; the OS releases its lease.
        return _expired(run)
    if child.stderr:
        print(child.stderr, file=sys.stderr, end="")
    try:
        result = json.loads(child.stdout)
    except ValueError as exc:
        raise RuntimeError("worker ended without a control receipt; inspect state and reconcile dispatched requests") from exc
    if not isinstance(result, dict) or "state" not in result:
        raise RuntimeError("invalid worker control receipt")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = None
    try:
        from run_pipeline import load_config
        from .controller import execute
        config = load_config(args.config)
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(args.run_dir, config)
    except (Exception, SystemExit) as exc:
        result = {"state": "FATAL_ERROR", "error": error_details(exc, config.LLM_SETTINGS if config else {})}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["state"] == "DONE" else 2 if result["state"] in {"PAUSED", "HUMAN_GATE", "HUMAN_GATE_SMOKE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
