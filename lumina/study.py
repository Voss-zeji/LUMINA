"""Human-facing study execution over the existing receipted runtime."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .agent.budget import Budget
from .agent.contracts import ResearchSpecification, validate_manifest
from .agent.controller import approve, begin
from .agent.store import Store
from .agent.worker import supervise
from .common import error_details


def preflight(config) -> ResearchSpecification:
    spec = ResearchSpecification.from_config(config.REQUEST, config)
    sources = {model['source'] for model in spec.data['models']} | {spec.data['embedding']['source']}
    for source in sources:
        key = config.LLM_SETTINGS[source].get('key')
        if not isinstance(key, str) or not key.strip() or key.startswith('REPLACE'):
            raise ValueError(f'Provider {source} needs a key in secrets.local.toml')
    return spec


def _summary(run: Path, result: dict) -> list[dict]:
    labels = {'DONE': 'Completed', 'HUMAN_GATE_SMOKE': 'Trial complete; review required',
              'PAUSED': 'Paused at the requested stage', 'RECOVERABLE_ERROR': 'Stopped; resolve the reported error',
              'HUMAN_GATE': 'Stopped; an explicit decision is required', 'FATAL_ERROR': 'Stopped; integrity check failed'}
    print(labels.get(result['state'], 'Research pipeline in progress'))
    print(f'Results: {run / "outputs"}')
    with Store(run) as store:
        spec = validate_manifest(run, store.snapshot()['run']['spec_hash'])['specification']
        totals = Budget(store, spec).totals()
        tasks = [t for t in store.tasks('examiner') if t['payload'].get('paper_uid')]
        completed = sum(t['status'] == 'succeeded' for t in tasks)
        failed = sum(t['status'] in {'failed', 'unresolved'} for t in tasks)
        print(f'Extraction tasks: {completed}/{len(tasks)} completed; {failed} failed or unresolved')
        print(f'Calls: {totals["calls"]}; known cost: {totals["known_cost"]}; reserved cost: {totals["held_cost"]}')
        pending = [g for g in store.gates() if g['status'] == 'pending']
        for gate in pending:
            print(f'Review: {gate["reason"]}')
    if result.get('error') or result.get('reason'):
        print(result.get('error') or result['reason'])
    print(f'Diagnostics: {run / "reports"}')
    for path in sorted((run / 'outputs' / 'prepared').glob('*.meta.json')):
        conversion = json.loads(path.read_text(encoding='utf-8')).get('converter')
        if isinstance(conversion, dict) and conversion.get('text_only'):
            print(f'Review PDF text conversion: {path.stem}; pypdf was used. Check table structure and reading order.')
    return pending


def run_study(config_path: str, config, *, check: bool = False) -> int:
    """One command creates or resumes exactly the configured run, never guesses."""
    try:
        spec = preflight(config)
        project = config.PROJECT
        root = Path(project['runs_dir']).resolve()
        run = root / project['run_id']
        if run.exists():
            with Store(run) as store:
                manifest = validate_manifest(run, store.snapshot()['run']['spec_hash'])
            if spec.fingerprint != manifest['specification_hash']:
                raise ValueError('Study configuration, papers or code changed; use the original revision/config or a new project.run_id')
        if check:
            data = spec.data
            print(f'Configuration valid: {len(data["papers"])} papers, {len(data["questions"])} questions, '
                  f'{len(data["models"])} models; no model requests sent')
            return 0
        if not run.exists():
            run = begin(config.REQUEST, config, root, dry_run=True, run_id=project['run_id'])
        result = supervise(run, config_path, progress=True)
        pending = _summary(run, result)
        if result['state'] == 'HUMAN_GATE_SMOKE':
            print(f'Review {run / "reports" / "smoke_report.json"} and original evidence, units and experiment identities.')
            if not sys.stdin.isatty():
                print('Waiting for confirmation. Run the same command in an interactive terminal to review and continue.')
                return 2
            smoke = [g for g in pending if g['kind'] == 'smoke']
            if len(smoke) != 1 or len(pending) != 1:
                raise ValueError('Additional unresolved decisions prevent batch approval; inspect diagnostics')
            try:
                answer = input('After reviewing the trial, continue the full batch? [y/N] ').strip().lower()
            except (EOFError, KeyboardInterrupt):
                print('\nTrial retained. Batch has not been approved.')
                return 2
            if answer not in {'y', 'yes'}:
                print('Trial retained. Batch has not been approved.')
                return 2
            # Bind approval to the frozen specification and the report just displayed.
            from .configuration import load_config
            fresh = load_config(config_path)
            if preflight(fresh).fingerprint != spec.fingerprint:
                raise ValueError('Configuration changed during review; batch was not approved')
            approve(run, smoke[0]['gate_id'], 'User reviewed trial evidence, units, experiments and costs in the study CLI', config=fresh)
            result = supervise(run, config_path, progress=True)
            _summary(run, result)
        return 0 if result['state'] == 'DONE' else 1 if result['state'] in {'FATAL_ERROR', 'RECOVERABLE_ERROR'} else 2
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(error_details(exc, config.LLM_SETTINGS), file=sys.stderr)
        return 1
