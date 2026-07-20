from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_config(config_path: str):
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config not found: {path}. Copy config.example.py to {path.name} and fill local paths/secrets first.")
    spec = importlib.util.spec_from_file_location("lumina_user_config", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_llm_dicts(config) -> dict:
    return {k: config.FULL_LLM_POOL[k] for k in config.SELECTED_KEYS if k in config.FULL_LLM_POOL}


def run(domain: str, stage: str, config) -> None:
    from lumina import composite, cross_validation, ensemble, examiner, preparation
    from lumina.utils import model_name

    if domain not in config.DOMAINS:
        raise SystemExit(f"unknown domain {domain}; expected one of {sorted(config.DOMAINS)}")
    domain_cfg = config.DOMAINS[domain]
    llm_dicts = build_llm_dicts(config)
    selected_model_names = [model_name(llm) for llm in llm_dicts.values()]
    n_questions = len(domain_cfg["questions"])

    if stage in ("prepare", "all"):
        mds = preparation.ensure_markdowns(domain_cfg)
        for row in preparation.token_audit([str(p) for p in mds]):
            print(row)
    if stage in ("examiner", "all"):
        examiner.run_examiner_for_domain(domain, domain_cfg, llm_dicts, config.LLM_SETTINGS, config.RUN)
    if stage in ("composite", "all"):
        composite.create_baseline_composite(domain_cfg, list(range(1, n_questions + 1)), models=selected_model_names)
    if stage in ("embeddings", "all"):
        cross_validation.generate_embeddings_for_domain(domain_cfg, config.EMBEDDING_MODEL, config.LLM_SETTINGS, config.RUN)
    if stage in ("cross", "all"):
        cross_validation.cross_validate_domain(
            domain,
            domain_cfg,
            llm_dicts,
            config.LLM_SETTINGS,
            config.EMBEDDING_MODEL,
            config.RUN,
            selected_model_names,
        )
        cross_validation.aggregate_cross_scores(domain_cfg, selected_model_names)
    if stage in ("ensemble", "all"):
        ensemble.run_ensemble_for_domain(domain, domain_cfg, config.RUN)


# ---- LUMINA Pipeline CLI ----
# Usage:
#   python run_pipeline.py --domain {aqua,wildfire} --stage {prepare|examiner|composite|embeddings|cross|ensemble|all}
#   python run_pipeline.py --config path/to/config.py --domain wildfire --stage all
#
# Stages:
#   0. prepare    → PDF→MD conversion, refs-truncation, token audit
#   1. examiner   → Multi-LLM JSON extraction per paper per question
#   2. composite  → Aggregate all model outputs into Excel
#   3. embeddings → Chunk + embed markdown, cache as .npy
#   4. cross      → Cross-validation: verify evidence via embedding retrieval
#   5. ensemble   → Consensus voting with cross_score threshold filtering
#   all           → Run stages 0→5 in sequence
def main() -> None:
    parser = argparse.ArgumentParser(description="LUMINA aqua/wildfire pipeline")
    parser.add_argument("--config", default="config.py", help="Path to a filled config.py (default: ./config.py)")
    parser.add_argument("--domain", required=True, choices=["aqua", "wildfire"])
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "examiner", "composite", "embeddings", "cross", "ensemble", "all"],
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run(args.domain, args.stage, config)


if __name__ == "__main__":
    main()
