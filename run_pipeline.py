"""Command entry; scientific functions remain importable for legacy callers."""
from __future__ import annotations

import argparse
from pathlib import Path

from lumina.configuration import load_config
from lumina.pipeline import (  # noqa: F401 - supported legacy import surface
    _run, build_llm_dicts, run, validate_composites, validate_min_cross_scores,
)


def main(argv=None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"run", "status", "pause", "resume", "approve", "report", "evaluate", "import"}:
        from lumina.agent.cli import main as agent_main
        raise SystemExit(agent_main(argv))
    parser = argparse.ArgumentParser(description="LUMINA research pipeline")
    parser.add_argument("--config", default="config.py", help="Study project.toml or legacy config.py")
    parser.add_argument("--domain", choices=["aqua", "wildfire"], help="Legacy Python-config domain")
    parser.add_argument("--check", action="store_true", help="Validate a TOML study without model calls")
    parser.add_argument(
        "--stage",
        required=False,
        choices=["prepare", "examiner", "composite", "embeddings", "cross", "ensemble", "all"],
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Configuration error: {exc}\n')
    if Path(args.config).suffix.lower() == ".toml":
        if args.domain or args.stage:
            parser.error("TOML studies take domain and stage from project.toml")
        from lumina.study import run_study
        raise SystemExit(run_study(args.config, config, check=args.check))
    if args.check or not args.domain or not args.stage:
        parser.error("legacy Python config requires --domain and --stage; --check requires TOML")
    run(args.domain, args.stage, config)


if __name__ == "__main__":
    main()
