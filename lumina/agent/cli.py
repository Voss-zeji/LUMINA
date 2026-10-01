"""CLI control operations; legacy flags are still handled by run_pipeline.py."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

from ..common import error_details
from .budget import Budget
from .contracts import validate_manifest
from .controller import approve, begin
from .lease import RunLease
from .report import write_report
from .store import Store


def _run_path(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must be a directory name, not a path")
    root = Path(root).resolve()
    run = (root / run_id).resolve()
    if not run.is_relative_to(root) or not run.is_dir():
        raise ValueError("run not found inside runs-dir")
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LUMINA persistent run control")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="freeze an explicit specification, optionally plan without executing")
    run.add_argument("--spec", required=True)
    run.add_argument("--config", default="config.py")
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--run-id")
    run.add_argument("--dry-run", action="store_true")
    for command in ("status", "pause", "resume", "approve", "report", "evaluate", "import"):
        sub = commands.add_parser(command)
        sub.add_argument("--run-id", required=True)
        sub.add_argument("--runs-dir", default="runs")
        if command == "resume":
            sub.add_argument("--config", default="config.py")
        elif command == "approve":
            sub.add_argument("--gate-id", required=True)
            sub.add_argument("--reason", required=True)
            sub.add_argument("--outcome", choices=["not_executed", "executed_unknown"])
        elif command == "evaluate":
            sub.add_argument("--gold")
            sub.add_argument("--output-dir")
        elif command == "import":
            sub.add_argument("--from-run-id", required=True)
            sub.add_argument("--task-id", required=True)
            sub.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    config = None
    try:
        if args.command in {"run", "resume"}:
            from run_pipeline import load_config
            config = load_config(args.config)
        if args.command == "run":
            request = json.loads(Path(args.spec).read_text(encoding="utf-8"))
            path = begin(request, config, args.runs_dir, dry_run=True, run_id=args.run_id)
            if args.dry_run:
                with Store(path) as store:
                    result = dict(run_id=path.name, run_dir=str(path), state=store.snapshot()["run"]["state"])
            else:
                from .worker import supervise
                result = supervise(path, args.config)
        else:
            path = _run_path(args.runs_dir, args.run_id)
            if args.command == "resume":
                from .worker import supervise
                result = supervise(path, args.config)
            elif args.command == "approve":
                result = approve(path, args.gate_id, args.reason, outcome=args.outcome)
            elif args.command == "evaluate":
                from ..evaluation import evaluate_run
                result = evaluate_run(path, gold=args.gold, output_dir=args.output_dir)
            elif args.command == "import":
                from .imports import import_task
                result = import_task(_run_path(args.runs_dir, args.from_run_id), path, args.task_id, reason=args.reason)
            elif args.command == "report":
                with RunLease(path), Store(path) as store:
                    manifest = validate_manifest(path, store.snapshot()["run"]["spec_hash"])
                    context = SimpleNamespace(run_dir=path, spec=manifest["specification"], store=store,
                                              budget=Budget(store, manifest["specification"]))
                    qc_file = path / "checkpoints" / "final_qc.json"
                    qc = json.loads(qc_file.read_text(encoding="utf-8")) if qc_file.exists() else None
                    result = write_report(context, "status_report", final_qc=qc)
            else:
                with Store(path) as store:
                    manifest = validate_manifest(path, store.snapshot()["run"]["spec_hash"])
                    if args.command == "pause":
                        if store.snapshot()["run"]["state"] in {"DONE", "FATAL_ERROR"}:
                            raise ValueError("terminal run cannot be paused")
                        store.request_pause()
                        store.export_views()
                    result = store.snapshot()
                    result["budget"] = Budget(store, manifest["specification"]).totals()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        state = result.get("state")
        if state in {"RECOVERABLE_ERROR", "FATAL_ERROR"}:
            return 1
        if state in {"HUMAN_GATE", "HUMAN_GATE_SMOKE", "PAUSED"}:
            return 2
        return 0
    except (Exception, SystemExit) as exc:
        settings = config.LLM_SETTINGS if config is not None else {}
        print(json.dumps(dict(status="failed", error=error_details(exc, settings)), ensure_ascii=False))
        return 1
