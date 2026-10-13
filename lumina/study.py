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
from .ui_support import atomic_text, redact


def preflight(config) -> ResearchSpecification:
    spec = ResearchSpecification.from_config(config.REQUEST, config)
    sources = {model['source'] for model in spec.data['models']} | {spec.data['embedding']['source']}
    for source in sources:
        key = config.LLM_SETTINGS[source].get('key')
        if not isinstance(key, str) or not key.strip() or key.startswith('REPLACE'):
            raise ValueError(f'Provider {source} needs a key in secrets.local.toml')
    return spec


def _summary(run: Path, result: dict, language='en') -> list[dict]:
    def text(zh, en):
        return zh if language == 'zh' else en
    labels = {'DONE': 'Completed', 'HUMAN_GATE_SMOKE': 'Trial complete; review required',
              'PAUSED': 'Paused at the requested stage', 'RECOVERABLE_ERROR': 'Stopped; resolve the reported error',
              'HUMAN_GATE': 'Stopped; an explicit decision is required', 'FATAL_ERROR': 'Stopped; integrity check failed'}
    chinese = {'DONE': '运行完成，科学结果仍需独立核查', 'HUMAN_GATE_SMOKE': '试跑完成，等待审核',
               'PAUSED': '运行已暂停', 'RECOVERABLE_ERROR': '运行停止，请处理错误后恢复',
               'HUMAN_GATE': '运行停止，需要人工处理', 'FATAL_ERROR': '完整性检查失败，运行停止'}
    print((chinese if language == 'zh' else labels).get(result['state'], text('科研流程运行中', 'Research pipeline in progress')))
    print(text(f'结果：{run / "outputs"}', f'Results: {run / "outputs"}'))
    with Store(run) as store:
        spec = validate_manifest(run, store.snapshot()['run']['spec_hash'])['specification']
        totals = Budget(store, spec).totals()
        tasks = [t for t in store.tasks('examiner') if t['payload'].get('paper_uid')]
        completed = sum(t['status'] == 'succeeded' for t in tasks)
        failed = sum(t['status'] in {'failed', 'unresolved'} for t in tasks)
        print(text(f'抽取任务：{completed}/{len(tasks)} 完成；{failed} 失败或未决',
                   f'Extraction tasks: {completed}/{len(tasks)} completed; {failed} failed or unresolved'))
        print(text(f'调用：{totals["calls"]}；已知 token：{totals["known_tokens"]}；预留 token：{totals["held_tokens"]}',
                   f'Calls: {totals["calls"]}; known tokens: {totals["known_tokens"]}; held tokens: {totals["held_tokens"]}'))
        unpriced = totals.get('unpriced_requests', 0)
        complete = totals.get('cost_complete', not unpriced and not totals.get('unknown_usage'))
        print(text(f'已计价部分（估算）：{totals["known_cost"]}；预留费用：{totals["held_cost"]}；未计价请求：{unpriced}',
                   f'Priced subtotal (estimate): {totals["known_cost"]}; priced reservation: {totals["held_cost"]}; unpriced requests: {unpriced}'))
        if not complete:
            print(text('总费用：未知或尚未完整结算', 'Total cost: unknown or not fully settled'))
        pending = [g for g in store.gates() if g['status'] == 'pending']
        for gate in pending:
            print(text(f'待处理：{gate["reason"]}', f'Review: {gate["reason"]}'))
    if result.get('error') or result.get('reason'):
        print(result.get('error') or result['reason'])
    print(text(f'日志与报告：{run / "reports"}', f'Diagnostics: {run / "reports"}'))
    for path in sorted((run / 'outputs' / 'prepared').glob('*.meta.json')):
        conversion = json.loads(path.read_text(encoding='utf-8')).get('converter')
        if isinstance(conversion, dict) and conversion.get('text_only'):
            print(text(f'请复核 PDF 转换：{path.stem} 使用了 pypdf，需检查表格和阅读顺序。',
                       f'Review PDF text conversion: {path.stem}; pypdf was used. Check table structure and reading order.'))
    return pending


def _link_setup(run: Path, directory: Path | None, config) -> None:
    if directory is None:
        return
    directory = Path(directory).resolve()
    # Copy only startup logs, never the draft directory or secrets.local.toml.
    secrets = [p.get('key', '') for p in config.LLM_SETTINGS.values()]
    destination = run / 'reports' / 'setup' / directory.name
    for name in ('events.jsonl', 'installation.log', 'startup.log', 'errors.log'):
        source = directory / name
        if source.is_file():
            atomic_text(destination / name, redact(source.read_text(encoding='utf-8', errors='replace'), secrets), private=True)


def run_study(config_path: str, config, *, check: bool = False, setup_log: Path | None = None) -> int:
    """One command creates or resumes exactly the configured run, never guesses."""
    try:
        language = getattr(config, 'PROJECT', {}).get('language', 'en')
        def text(zh, en):
            return zh if language == 'zh' else en
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
            print(text(f'配置校验通过：{len(data["papers"])} 篇文献，{len(data["questions"])} 个问题，{len(data["models"])} 个模型；未发出模型请求',
                       f'Configuration valid: {len(data["papers"])} papers, {len(data["questions"])} questions, {len(data["models"])} models; no model requests sent'))
            return 0
        if not run.exists():
            run = begin(config.REQUEST, config, root, dry_run=True, run_id=project['run_id'])
        _link_setup(run, setup_log, config)
        result = supervise(run, config_path, progress=True, language=language)
        pending = _summary(run, result, language)
        if result['state'] == 'HUMAN_GATE_SMOKE':
            print(text(f'请检查 {run / "reports" / "smoke_report.json"} 与原文证据、单位和实验归属。',
                       f'Review {run / "reports" / "smoke_report.json"} and original evidence, units and experiment identities.'))
            if not sys.stdin.isatty():
                print(text('等待确认。请在交互终端重新执行同一命令，审核后继续。',
                           'Waiting for confirmation. Run the same command in an interactive terminal to review and continue.'))
                return 2
            smoke = [g for g in pending if g['kind'] == 'smoke']
            if len(smoke) != 1 or len(pending) != 1:
                raise ValueError('Additional unresolved decisions prevent batch approval; inspect diagnostics')
            try:
                answer = input(text('审核试跑结果后，是否继续整批？[y/N] ', 'After reviewing the trial, continue the full batch? [y/N] ')).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print(text('\n试跑记录已保留，整批未获批准。', '\nTrial retained. Batch has not been approved.'))
                return 2
            if answer not in {'y', 'yes'}:
                print(text('试跑记录已保留，整批未获批准。', 'Trial retained. Batch has not been approved.'))
                return 2
            # Bind approval to the frozen specification and the report just displayed.
            from .configuration import load_config
            fresh = load_config(config_path)
            if preflight(fresh).fingerprint != spec.fingerprint:
                raise ValueError('Configuration changed during review; batch was not approved')
            approve(run, smoke[0]['gate_id'], 'User reviewed trial evidence, units, experiments and costs in the study CLI', config=fresh)
            result = supervise(run, config_path, progress=True, language=language)
            _summary(run, result, language)
        _link_setup(run, setup_log, config)
        return 0 if result['state'] == 'DONE' else 1 if result['state'] in {'FATAL_ERROR', 'RECOVERABLE_ERROR'} else 2
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(error_details(exc, config.LLM_SETTINGS), file=sys.stderr)
        return 1
