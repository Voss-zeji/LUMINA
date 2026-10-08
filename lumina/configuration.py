"""Centralized TOML study loading and legacy Python configuration compatibility."""
from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

from . import prompts

STAGES = {'all', 'prepare', 'examiner', 'composite', 'embeddings', 'cross', 'ensemble'}


def _table(value, allowed: set[str], required: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise ValueError(f'{label}: unknown or missing fields; required {sorted(required)}, allowed {sorted(allowed)}')
    return value


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or 'REPLACE' in value:
        raise ValueError(f'{label} must be filled with nonempty text, not a placeholder')
    return value


def _read(path: Path) -> dict:
    with path.open('rb') as stream:
        return tomllib.load(stream)


def load_config(config_path: str):
    path = Path(config_path).resolve()
    if path.suffix.lower() == '.py':
        if not path.is_file():
            raise SystemExit(f'Config not found: {path}. Copy config.example.py and fill local paths/secrets first.')
        spec = importlib.util.spec_from_file_location('lumina_user_config', path)
        if spec is None or spec.loader is None:
            raise ValueError('Cannot load Python config')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if path.suffix.lower() != '.toml':
        raise ValueError('Configuration must be project.toml or a legacy .py file')
    raw = _read(path)
    sections = {'project', 'selected_models', 'models', 'providers', 'embedding', 'run', 'budget', 'pricing'}
    _table(raw, sections, sections, 'project file')
    fields = {'domain', 'domain_knowledge', 'run_id', 'runs_dir', 'papers', 'questions', 'secrets',
              'rounds', 'smoke_size', 'stage', 'allow_pdf_resources'}
    project = _table(raw['project'], fields, {'domain', 'domain_knowledge', 'run_id', 'runs_dir'}, 'project')
    domain = _text(project['domain'], 'project.domain')
    knowledge = _text(project['domain_knowledge'], 'project.domain_knowledge')
    run_id = _text(project['run_id'], 'project.run_id')
    from .agent.contracts import RESERVED_NAMES, ResearchSpecification
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id) or run_id.lower() in RESERVED_NAMES:
        raise ValueError('project.run_id must be a portable directory name')
    stage = project.get('stage', 'all')
    if not isinstance(stage, str) or stage not in STAGES:
        raise ValueError(f'project.stage must be one of {sorted(STAGES)}')

    def relative(value, label):
        return (path.parent / _text(value, label)).resolve()

    runs_dir = relative(project['runs_dir'], 'project.runs_dir')
    question_path = relative(project.get('questions', 'questions.toml'), 'project.questions')
    question_set = prompts.validate_definition(_read(question_path))
    paper_list = relative(project.get('papers', 'papers.txt'), 'project.papers')
    papers = [str(relative(line.strip(), 'paper path')) for line in paper_list.read_text(encoding='utf-8-sig').splitlines()
              if line.strip() and not line.lstrip().startswith('#')]
    secrets_path = relative(project.get('secrets', 'secrets.local.toml'), 'project.secrets')
    secrets = _table(_read(secrets_path), {'providers'}, {'providers'}, 'secrets')
    providers = raw['providers']
    if not isinstance(providers, dict) or not providers:
        raise ValueError('providers must be a nonempty table')
    secret_providers = _table(secrets['providers'], set(providers), set(), 'secret providers')
    settings = {}
    for source, provider in providers.items():
        _table(provider, {'url', 'supports_json_mode'}, {'url'}, f'providers.{source}')
        secret = _table(secret_providers.get(source, {}), {'key'}, set(), f'secrets provider {source}')
        if 'key' in secret and not isinstance(secret['key'], str):
            raise ValueError('provider key must be text')
        settings[source] = dict(provider, key=secret.get('key', ''))
    models = raw['models']
    if not isinstance(models, dict) or not models:
        raise ValueError('models must be a nonempty table')
    for alias, model in models.items():
        _table(model, {'model', 'source', 'limit_token', 'timeout_seconds', 'rate_limit_seconds'},
               {'model', 'source'}, f'models.{alias}')
        _text(model['model'], 'model ID')
    selected = raw['selected_models']
    if not isinstance(selected, list) or not selected or any(not isinstance(k, str) for k in selected):
        raise ValueError('selected_models must be a nonempty string list')
    embedding = _table(raw['embedding'], {'model', 'source', 'url', 'timeout_seconds', 'rate_limit_seconds'},
                       {'model', 'source', 'url'}, 'embedding')
    _text(embedding['model'], 'embedding.model')
    run = raw['run']
    if not isinstance(run, dict):
        raise ValueError('run must be a table')
    request = dict(domain=domain, papers=papers, rounds=project.get('rounds', [run.get('round_index')]),
                   smoke_size=project.get('smoke_size', 1), budget=raw['budget'], pricing=raw['pricing'],
                   allow_pdf_resources=project.get('allow_pdf_resources', False), advisor=None)
    cfg = dict(domain_knowledge=knowledge, questions=list(range(1, len(question_set['questions']) + 1)),
               question_set=question_set)
    config = SimpleNamespace(DOMAINS={domain: cfg}, FULL_LLM_POOL=models, SELECTED_KEYS=selected,
                             LLM_SETTINGS=settings, EMBEDDING_MODEL=embedding, RUN=run, REQUEST=request,
                             PROJECT=dict(run_id=run_id, runs_dir=str(runs_dir), stage=stage))
    # Reuse the pipeline's authority for paper identity, prices, budgets and model validation.
    ResearchSpecification.from_config(request, config)
    return config
