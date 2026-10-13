"""CLI worker supervision: the approved wall deadline also bounds blocked local work."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from ..common import error_details, save_json
from .budget import Budget
from .contracts import validate_manifest
from .lease import RunLease
from .store import Store
from ..progress import ProgressMonitor, language_for


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


def supervise(run_dir, config_path, *, worker_command=None, progress=False, language=None):
    run = Path(run_dir).resolve()
    with Store(run) as store:
        manifest = validate_manifest(run, store.snapshot()["run"]["spec_hash"])
        if store.snapshot()["run"]["state"] in {"DONE", "FATAL_ERROR"}:
            from .controller import execute
            from run_pipeline import load_config
            return execute(run, load_config(str(config_path)))
        max_runtime = manifest["specification"]["budget"].get("max_runtime")
        used = Budget(store, manifest["specification"]).totals().get("runtime", 0)
        remaining = None if max_runtime is None else max_runtime - used
    if remaining is not None and remaining <= 0:
        return _expired(run)
    command = worker_command or [sys.executable, "-m", "lumina.agent.worker", "--run-dir", str(run),
                                 "--config", str(Path(config_path).resolve())]
    if progress:
        return _supervise_with_progress(run, command, remaining, config_path, language=language)
    try:
        child = subprocess.run(command, stdout=subprocess.PIPE, stderr=None if progress else subprocess.PIPE,
                               text=True, encoding="utf-8", timeout=remaining,
                               cwd=Path(__file__).resolve().parents[2],
                               env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
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


def _progress_language(config_path, language=None) -> str:
    if language in {"zh", "en"}:
        return language
    try:
        from run_pipeline import load_config
        return language_for(load_config(str(config_path)))
    except (Exception, SystemExit):
        return "en"


def _pause_after_interrupt(run: Path) -> dict:
    """Preserve the ledger and make dispatched work explicitly unreplayable."""
    with RunLease(run), Store(run) as store:
        manifest = validate_manifest(run, store.snapshot()["run"]["spec_hash"])
        budget = Budget(store, manifest["specification"])
        recovery = budget.recover_unfinished()
        for attempt in recovery["marked_unknown"]:
            if not any(g["kind"] == "unknown_request" and g["target"] == attempt and g["status"] == "pending"
                       for g in store.gates()):
                store.create_gate("unknown_request", attempt, "supervisor interrupted after dispatch; execution outcome unknown")
        state = store.snapshot()["run"]["state"]
        if state not in {"DONE", "FATAL_ERROR", "PAUSED"}:
            store.request_pause()
            store.set_state("PAUSED", "Ctrl-C: no promise to cancel an already dispatched request")
        store.export_views()
        return {"state": "PAUSED", "run_id": run.name, "reason": "Ctrl-C", "budget": budget.totals(),
                "scientific_validation": "NOT_EVALUATED"}


def _supervise_with_progress(run: Path, command, remaining, config_path, *, language=None) -> dict:
    """Supervise the same child receipt while a read-only monitor renders stderr UI."""
    monitor = ProgressMonitor(run, language=_progress_language(config_path, language), stream=sys.stderr)
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8",
                             cwd=Path(__file__).resolve().parents[2], env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    stop_observer = threading.Event()

    def observe():
        while not stop_observer.is_set():
            try:
                monitor.emit()
            except Exception:
                pass
            if stop_observer.wait(1.0):
                break
        try:
            monitor.emit()
        except Exception:
            pass

    observer = threading.Thread(target=observe, name="lumina-progress", daemon=True)
    observer.start()
    timed_out = interrupted = False
    try:
        try:
            stdout, _ = child.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, _ = child.communicate()
            timed_out = True
    except KeyboardInterrupt:
        try:
            child.terminate()
        except OSError:
            pass
        try:
            stdout, _ = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, _ = child.communicate()
        interrupted = True
    finally:
        stop_observer.set()
        observer.join(timeout=0.25)
    if timed_out:
        return _expired(run)
    if interrupted:
        return _pause_after_interrupt(run)
    try:
        result = json.loads(stdout)
    except ValueError as exc:
        raise RuntimeError("worker ended without a control receipt; inspect state and reconcile dispatched requests") from exc
    if not isinstance(result, dict) or "state" not in result:
        raise RuntimeError("invalid worker control receipt")
    # stdout is parsed exactly as in the non-progress path; progress never writes to it.
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
        with contextlib.ExitStack() as stack:
            output = sys.stderr
            if hasattr(config, 'PROJECT'):
                output = stack.enter_context((Path(args.run_dir) / 'reports' / 'pipeline.log').open('a', encoding='utf-8'))
            with contextlib.redirect_stdout(output):
                result = execute(args.run_dir, config)
    except (Exception, SystemExit) as exc:
        result = {"state": "FATAL_ERROR", "error": error_details(exc, config.LLM_SETTINGS if config else {})}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["state"] == "DONE" else 2 if result["state"] in {"PAUSED", "HUMAN_GATE", "HUMAN_GATE_SMOKE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
