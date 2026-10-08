"""Shared scientific pipeline for study and legacy command interfaces."""
from __future__ import annotations

from pathlib import Path


def build_llm_dicts(config) -> dict:
    unknown = set(config.SELECTED_KEYS) - set(config.FULL_LLM_POOL)
    if unknown:
        raise ValueError(f"unknown SELECTED_KEYS: {sorted(unknown)}")
    if not config.SELECTED_KEYS:
        raise ValueError("SELECTED_KEYS must not be empty")
    result = {k: config.FULL_LLM_POOL[k] for k in config.SELECTED_KEYS}
    names = [v['model'].split('/')[-1].lower() for v in result.values()]
    if len(names) != len(set(names)):
        raise ValueError("selected models have colliding model names; use unique model identities")
    return result


def validate_min_cross_scores(selected_model_names: list[str], run_cfg: dict) -> None:
    """Reject thresholds that no independent verifier set can reach."""
    max_score = max(len(set(selected_model_names)) - 1, 0)
    scores = run_cfg.get("min_cross_scores", [])
    invalid = [score for score in scores if type(score) is not int or not 1 <= score <= max_score]
    if invalid:
        raise SystemExit(
            f"min_cross_scores must be integers from 1 to {max_score} (at most {max_score} independent verifiers); "
            f"invalid: {invalid}"
        )


def run(domain: str, stage: str, config) -> None:
    from lumina.common import error_details, pipeline_lock, save_json

    if domain not in config.DOMAINS:
        raise SystemExit(f"unknown domain {domain}; expected one of {sorted(config.DOMAINS)}")
    cfg = config.DOMAINS[domain]
    lock = Path(cfg['examiner_output']).parent / f'.lumina-{domain}.lock'
    status = Path(cfg['ensemble_dir']) / 'last_run.json'
    with pipeline_lock(lock):
        try:
            _run(domain, stage, config)
        except (Exception, SystemExit) as exc:
            save_json(status, dict(status='failed', stage=stage, error=error_details(exc, config.LLM_SETTINGS)))
            raise
        else:
            save_json(status, dict(status='succeeded', stage=stage, round_index=config.RUN['round_index']))


def _run(domain: str, stage: str, config, runtime=None) -> None:
    from lumina import composite, cross_validation, ensemble, examiner, preparation
    from lumina.utils import model_name

    if domain not in config.DOMAINS:
        raise SystemExit(f"unknown domain {domain}; expected one of {sorted(config.DOMAINS)}")
    domain_cfg = config.DOMAINS[domain]
    llm_dicts = build_llm_dicts(config)
    selected_model_names = [model_name(llm) for llm in llm_dicts.values()]
    from lumina import prompts
    n_questions = len(prompts.questions_for_domain(domain, domain_cfg))
    runtime_args = {} if runtime is None else {"runtime": runtime}
    if domain_cfg["questions"] != list(range(1, n_questions + 1)):
        raise ValueError("question indices must match the configured question list")
    if type(config.RUN.get('round_index')) is not int or config.RUN['round_index'] < 1:
        raise ValueError('round_index must be a positive integer')
    if stage in ("cross", "ensemble", "all"):
        validate_min_cross_scores(selected_model_names, config.RUN)
        ensemble.consensus_threshold(selected_model_names, config.RUN)

    if stage in ("prepare", "all"):
        mds = preparation.ensure_markdowns(domain_cfg)
        from lumina.common import paper_markdowns
        paper_markdowns(domain_cfg['markdown_dir'])
        for row in preparation.token_audit([str(p) for p in mds]):
            print(row)
    if stage in ("examiner", "all"):
        examiner.run_examiner_for_domain(domain, domain_cfg, llm_dicts, config.LLM_SETTINGS, config.RUN, **runtime_args)
    if stage in ("composite", "all"):
        composite.create_baseline_composite(domain_cfg, list(range(1, n_questions + 1)), models=selected_model_names,
            round_index=config.RUN['round_index'], domain=domain,
            expected_tasks=examiner.expected_tasks(domain, domain_cfg, llm_dicts, config.LLM_SETTINGS, config.RUN))
    if stage in ("embeddings", "all"):
        cross_validation.generate_embeddings_for_domain(domain_cfg, config.EMBEDDING_MODEL, config.LLM_SETTINGS, config.RUN, **runtime_args)
    if stage in ("cross", "all"):
        validate_composites(domain, domain_cfg, llm_dicts, config.LLM_SETTINGS, config.RUN)
        cross_validation.cross_validate_domain(
            domain,
            domain_cfg,
            llm_dicts,
            config.LLM_SETTINGS,
            config.EMBEDDING_MODEL,
            config.RUN,
            selected_model_names,
            **runtime_args,
        )
        cross_validation.aggregate_cross_scores(domain_cfg, selected_model_names,
            expected_verifiers=cross_validation.verifier_signatures(llm_dicts, config.LLM_SETTINGS, config.EMBEDDING_MODEL, config.RUN, domain_cfg))
    if stage in ("ensemble", "all"):
        if stage == 'ensemble':
            validate_composites(domain, domain_cfg, llm_dicts, config.LLM_SETTINGS, config.RUN)
            cross_validation.aggregate_cross_scores(domain_cfg, selected_model_names,
                expected_verifiers=cross_validation.verifier_signatures(llm_dicts, config.LLM_SETTINGS, config.EMBEDDING_MODEL, config.RUN, domain_cfg))
        ensemble.run_ensemble_for_domain(domain, domain_cfg, config.RUN, selected_model_names)


def validate_composites(domain: str, domain_cfg: dict, models: dict, settings: dict, run_cfg: dict) -> None:
    from lumina import examiner, prompts
    from lumina.common import read_composite

    tasks = examiner.expected_tasks(domain, domain_cfg, models, settings, run_cfg)
    expected = {(m['paper_index'], m['question_index'], m['model']): m for m in tasks.values()}
    found = set()
    for q in range(1, len(prompts.questions_for_domain(domain, domain_cfg)) + 1):
        file = Path(domain_cfg['composite_dir']) / f"{domain_cfg['composite_prefix']}_Q{q:02d}.xlsx"
        frame = read_composite(file)
        for _, row in frame.iterrows():
            key = (row['paper_index'], row['question_index'], row['model'])
            meta = expected.get(key)
            if meta is None or any(str(row[field]) != str(meta[field]) for field in ['request_fingerprint', 'paper_fingerprint', 'round_index']):
                raise ValueError(f'stale composite {file}: {key}; rerun examiner/composite for the selected round and configuration')
            found.add(key)
    if found != set(expected):
        raise ValueError('incomplete composite coverage; rerun examiner/composite')
