"""Bilingual, configuration-backed terminal setup. No scientific requests while editing."""
from __future__ import annotations

import argparse
import copy
import getpass
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import tomlkit

from . import prompts
from .configuration import load_config
from .pricing import read_quotes
from .ui_support import SetupLog, atomic_text, read_preferences, redact, save_preferences

ROOT = Path(__file__).resolve().parents[1]
BANNER = '''+--------------------------------------------------+
|                  L U M I N A                     |
| PDF -> Extraction -> Verification -> Consensus   |
+--------------------------------------------------+'''


def plain(value):
    return value.unwrap() if hasattr(value, 'unwrap') else value


def default(value):
    value = plain(value)
    return None if isinstance(value, str) and 'REPLACE' in value else value


def optional(table, name, value):
    if value is None or value == '':
        table.pop(name, None)
    else:
        table[name] = value


class Console:
    def __init__(self, language: str, log: SetupLog):
        self.language, self.log = language, log

    def text(self, zh, en):
        return zh if self.language == 'zh' else en

    def say(self, zh, en=None):
        print(redact(self.text(zh, en if en is not None else zh), self.log.secrets), flush=True)

    def ask(self, zh, en, value=None, *, required=False):
        value = default(value)
        label = self.text(zh, en)
        shown = str(value) if value not in (None, '') else self.text('必填' if required else '未设置', 'required' if required else 'not set')
        while True:
            response = input(redact(f'{label} [{shown}]: ', self.log.secrets)).strip()
            result = str(value) if not response and value is not None else response
            if required and not result:
                self.say('此项必须填写。', 'This field is required.')
                continue
            self.log.record('input', field=en, value=result)
            return result

    def number(self, zh, en, value=None, *, integer=False, minimum=0, required=False, positive=False):
        while True:
            raw = self.ask(zh + '（- 清空）', en + ' (- to clear)', value)
            if raw in {'', '-'} and not required:
                return None
            try:
                number = int(raw) if integer else float(raw)
                if not math.isfinite(number) or number < minimum or (positive and number == minimum):
                    raise ValueError
                return number
            except ValueError:
                self.say('请输入有效范围内的数字。', 'Enter a number within the allowed range.')

    def choice(self, zh, en, choices, value='1'):
        while True:
            raw = self.ask(zh, en, value)
            if raw in choices:
                return raw
            self.say('请选择列出的选项。', 'Choose one of the listed options.')

    def yes(self, zh, en, value=False):
        return self.choice(zh + ' (y/n)', en + ' (y/n)', {'y', 'n'}, 'y' if value else 'n') == 'y'

    def credential(self, provider, existing=''):
        existing = default(existing) or ''
        self.log.protect(existing)
        shown = self.text('已设置，回车保留', 'set; Enter to keep') if existing else self.text('必填', 'required')
        while True:
            value = getpass.getpass(f'{provider} API key [{shown}]: ').strip() or existing
            if value:
                self.log.protect(value)
                self.log.record('credential_set', provider=provider, configured=True)
                return value
            self.say('密钥不能为空。', 'A key is required.')


class StudyDraft:
    def __init__(self, target: Path, project, questions, secrets, papers):
        self.target = target.resolve()
        self.project, self.questions, self.secrets, self.papers = project, questions, secrets, papers
        self.base = self.target.parent

    @classmethod
    def create(cls, target: Path, template: str):
        folder = ROOT / 'configs' / template
        project = tomlkit.parse((folder / 'project.toml').read_text(encoding='utf-8'))
        questions = tomlkit.parse((folder / 'questions.toml').read_text(encoding='utf-8'))
        secrets = tomlkit.parse((folder / 'secrets.example.toml').read_text(encoding='utf-8'))
        project['models'] = tomlkit.table()
        project['selected_models'] = []
        project['budget'] = tomlkit.table()
        project['pricing'] = tomlkit.table()
        project['project']['run_id'] = 'study-' + datetime.now().strftime('%Y%m%d-%H%M%S')
        project['run'].pop('min_consensus_models', None)
        return cls(target, project, questions, secrets, [])

    @classmethod
    def load(cls, path: Path):
        path = path.expanduser().resolve(strict=True)
        project = tomlkit.parse(path.read_text(encoding='utf-8'))
        settings = project['project']
        def related(name, fallback):
            return (path.parent / str(settings.get(name, fallback))).resolve()
        questions = tomlkit.parse(related('questions', 'questions.toml').read_text(encoding='utf-8'))
        secret_path = related('secrets', 'secrets.local.toml')
        secrets = tomlkit.parse(secret_path.read_text(encoding='utf-8')) if secret_path.exists() else tomlkit.document()
        paper_path = related('papers', 'papers.txt')
        papers = []
        if paper_path.exists():
            papers = [str((path.parent / line.strip()).resolve()) for line in paper_path.read_text(encoding='utf-8-sig').splitlines()
                      if line.strip() and not line.lstrip().startswith('#') and 'REPLACE' not in line]
        return cls(path, project, questions, secrets, papers)

    def write(self, destination: Path, *, draft=False):
        """Write a self-contained four-file study; keep user TOML comments and values."""
        destination = destination.resolve()
        project = copy.deepcopy(self.project)
        settings = project['project']
        settings.update(papers='papers.txt', questions='questions.toml', secrets='secrets.local.toml')
        # Preserve effective output location when saving a copy or a startup draft.
        if destination.parent != self.base:
            settings['runs_dir'] = str((self.base / str(settings.get('runs_dir', '../../runs'))).resolve())
        text = tomlkit.dumps(project)
        if draft:
            text = '# Incomplete setup draft; validate before running.\n' + text
        folder = destination.parent
        files = {destination: text, folder / 'questions.toml': tomlkit.dumps(self.questions),
                 folder / 'secrets.local.toml': tomlkit.dumps(self.secrets),
                 folder / 'papers.txt': '\n'.join(self.papers) + '\n'}
        originals = {p: p.read_text(encoding='utf-8') if p.exists() else None for p in files}
        written = []
        try:
            for path, content in files.items():
                atomic_text(path, content, private=path.name == 'secrets.local.toml')
                written.append(path)
        except OSError:
            for path in reversed(written):
                if originals[path] is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_text(path, originals[path], private=path.name == 'secrets.local.toml')
            raise
        return destination

    def validate(self):
        with tempfile.TemporaryDirectory(prefix='lumina-check-') as temporary:
            path = self.write(Path(temporary) / 'project.toml')
            config = load_config(str(path))
            from .study import preflight
            return preflight(config)


def select_papers(ui: Console, current: list[str]) -> list[str]:
    ui.say(f'当前文献：{len(current)} 篇', f'Current papers: {len(current)}')
    method = ui.choice('1扫描目录 / 2导入 papers.txt / 3保留清单',
                       '1 scan folder / 2 import papers.txt / 3 keep list', {'1', '2', '3'}, '3' if current else '1')
    if method == '3':
        if not current:
            raise ValueError('Paper list is empty')
        return current
    path = Path(ui.ask('目录或清单路径', 'Folder or manifest path', required=True).strip('"')).expanduser().resolve(strict=True)
    if method == '1':
        if not path.is_dir():
            raise ValueError('Expected a PDF directory')
        recursive = ui.yes('包含子目录？', 'Include subfolders?')
        candidates = path.rglob('*') if recursive else path.iterdir()
        papers = sorted({str(p.resolve()) for p in candidates if p.is_file() and p.suffix.lower() == '.pdf'})
    else:
        papers = [str((path.parent / row.strip()).resolve(strict=True)) for row in path.read_text(encoding='utf-8-sig').splitlines()
                  if row.strip() and not row.lstrip().startswith('#')]
    if not papers:
        raise ValueError('No input papers found')
    for index, paper in enumerate(papers, 1):
        ui.say(f'{index}. {paper}')
    exclude = ui.ask('排除编号，以逗号分隔；回车保留全部', 'Exclude numbers separated by commas; Enter keeps all')
    excluded = {int(v.strip()) for v in exclude.split(',') if v.strip()}
    if any(i < 1 or i > len(papers) for i in excluded):
        raise ValueError('Excluded paper number is outside the list')
    papers = [p for i, p in enumerate(papers, 1) if i not in excluded]
    if not papers or not ui.yes(f'确认纳入 {len(papers)} 篇？', f'Use these {len(papers)} papers?', True):
        raise ValueError('Paper selection was not confirmed')
    return papers


def edit_questions(ui: Console, draft: StudyDraft):
    pending_edit = None
    while True:
        definition = prompts.validate_definition(draft.questions.unwrap())
        for q in definition['questions']:
            ui.say(f"{q['id']} [{q['kind']}] items={q['items'] or '*'} "
                   f"unit={q['require_unit']} experimental={q['require_experimental']}\n  {q['prompt'].strip()[:180]}")
        action = ui.choice('提示词：1保留 / 2查看全文 / 3编辑 / 4导入文件',
                           'Prompts: 1 keep / 2 full text / 3 edit / 4 import file', {'1', '2', '3', '4'})
        if action == '1':
            return
        if action == '2':
            ui.say(tomlkit.dumps(draft.questions))
            continue
        if action == '4':
            path = Path(ui.ask('questions.toml 路径', 'questions.toml path', required=True).strip('"')).expanduser()
        else:
            path = ui.log.directory / 'editor' / 'questions.toml'
            if pending_edit != path:
                atomic_text(path, tomlkit.dumps(draft.questions))
            editor = os.environ.get('VISUAL') or os.environ.get('EDITOR')
            command = shlex.split(editor, posix=os.name != 'nt') if editor else (['notepad.exe'] if os.name == 'nt' else [])
            command = [part.strip('"') for part in command]
            if not command:
                available = next((shutil.which(name) for name in ('nano', 'vi') if shutil.which(name)), None)
                command = [available] if available else []
            if command:
                ui.log.record('editor_opened', file=str(path), command=command)
                try:
                    subprocess.run([*command, str(path)], check=False)
                except OSError as exc:
                    ui.say(f'编辑器启动失败：{exc}', f'Editor could not start: {exc}')
                    ui.log.record('editor_failed', error=str(exc))
            ui.say(f'编辑文件：{path}', f'Edit file: {path}')
            input(ui.text('保存后按回车重新校验：', 'Save, then press Enter to validate: '))
        try:
            candidate = tomlkit.parse(path.read_text(encoding='utf-8'))
            prompts.validate_definition(candidate.unwrap())
        except (ValueError, OSError, TypeError) as exc:
            pending_edit = path if action == '3' else None
            ui.say(f'校验失败，保留原提示词：{exc}', f'Validation failed; original prompts retained: {exc}')
            ui.log.record('prompt_validation_failed', error=str(exc))
            continue
        draft.questions = candidate
        pending_edit = None
        ui.log.record('prompts_updated', source=str(path.resolve()), definition=candidate.unwrap())


def provider(ui: Console, draft: StudyDraft, alias: str, *, url=None):
    providers = draft.project.setdefault('providers', tomlkit.table())
    settings = providers.setdefault(alias, tomlkit.table())
    secrets = draft.secrets.setdefault('providers', tomlkit.table()).setdefault(alias, tomlkit.table())
    settings['url'] = ui.ask(f'{alias} 接口地址', f'{alias} endpoint', settings.get('url', url), required=True)
    from .agent.contracts import endpoint
    endpoint(str(settings['url']))
    secrets['key'] = ui.credential(alias, secrets.get('key', ''))
    return settings


def model_advanced(ui: Console, model, legacy=None, *, embedding=False):
    legacy = legacy or {}
    values = dict(max_input_tokens=default(model.get('max_input_tokens', legacy.get('max_input_tokens'))),
                  max_output_tokens=default(model.get('max_output_tokens', legacy.get('max_output_tokens', 0 if embedding else 4096))),
                  timeout_seconds=model.get('timeout_seconds', 120), rate_limit_seconds=model.get('rate_limit_seconds', 1))
    ui.say(f'高级请求参数：{values}', f'Advanced request settings: {values}')
    if not ui.yes('修改这些参数？', 'Edit these settings?'):
        return
    optional(model, 'max_input_tokens', ui.number('输入 token 上限', 'Input token ceiling', values['max_input_tokens'], integer=True, positive=True))
    model['max_output_tokens'] = ui.number('输出 token 上限', 'Output token ceiling', values['max_output_tokens'], integer=True, required=True, positive=not embedding)
    model['timeout_seconds'] = ui.number('单请求超时（秒）', 'Request timeout (seconds)', values['timeout_seconds'], required=True, positive=True)
    model['rate_limit_seconds'] = ui.number('请求间隔（秒）', 'Request interval (seconds)', values['rate_limit_seconds'], required=True)


def study_group(ui: Console, draft: StudyDraft):
    project = draft.project['project']
    project['domain'] = ui.ask('研究标识（结果文件名前缀）', 'Study identifier (output prefix)', project.get('domain', 'generic'), required=True)
    project['run_id'] = ui.ask('运行标识', 'Run identifier', project.get('run_id'), required=True)
    project['runs_dir'] = ui.ask('结果根目录（相对配置目录）', 'Results root (relative to config folder)', project.get('runs_dir', '../../runs'), required=True)
    project['domain_knowledge'] = ui.ask('研究背景与领域描述', 'Research/domain description', project.get('domain_knowledge'), required=True)


def extraction_group(ui: Console, draft: StudyDraft):
    draft.papers = select_papers(ui, draft.papers)
    models = draft.project.setdefault('models', tomlkit.table())
    current_keys = list(draft.project.get('selected_models', []))
    current = [str(models[k]['model']) for k in current_keys if k in models and default(models[k].get('model'))]
    names = [s.strip() for s in ui.ask('抽取模型完整 ID，以逗号分隔（至少两个）',
                                      'Extraction model IDs, comma separated (at least two)', ','.join(current), required=True).split(',') if s.strip()]
    if len(names) < 2 or len({n.split('/')[-1].lower() for n in names}) != len(names):
        raise ValueError('At least two distinct generating model identities are required')
    by_name = {str(v.get('model')): k for k, v in models.items()}
    selected = []
    for index, name in enumerate(names, 1):
        key = by_name.get(name, f'model_{index}')
        while key in selected or (key in models and str(models[key].get('model')) != name):
            key += '_new'
        model = models.setdefault(key, tomlkit.table())
        model['model'] = name
        model['source'] = ui.ask(f'{name} 服务商代号（可复用）', f'{name} provider alias (reusable)', model.get('source', 'chat'), required=True)
        settings = provider(ui, draft, str(model['source']))
        settings['supports_json_mode'] = ui.yes('接口支持 JSON mode？', 'Provider supports JSON mode?', bool(plain(settings.get('supports_json_mode', True))))
        model_advanced(ui, model, draft.project.get('pricing', {}).get(name))
        selected.append(key)
        draft.project['selected_models'] = selected[:]
    edit_questions(ui, draft)
    pdf = draft.project.setdefault('pdf', tomlkit.table())
    ui.say(f'PDF: backend={pdf.get("backend", "marker")}, fallback={pdf.get("fallback", "pypdf")}; '
           f'allow_pdf_resources={draft.project["project"].get("allow_pdf_resources", True)}')
    if ui.yes('修改 PDF 转换设置？', 'Edit PDF conversion settings?'):
        pdf['backend'] = ui.choice('PDF 工具', 'PDF backend', {'marker', 'pypdf'}, str(pdf.get('backend', 'marker')))
        pdf['fallback'] = ui.choice('后备工具', 'Fallback', {'pypdf', 'none'}, 'none' if pdf['backend'] == 'pypdf' else str(pdf.get('fallback', 'pypdf')))
        draft.project['project']['allow_pdf_resources'] = ui.yes('允许 Marker 准备或下载模型？', 'Allow Marker model preparation/download?', bool(plain(draft.project['project'].get('allow_pdf_resources', True))))


def embedding_group(ui: Console, draft: StudyDraft):
    model = draft.project.setdefault('embedding', tomlkit.table())
    model['model'] = ui.ask('Embedding 模型 ID', 'Embedding model ID', model.get('model'), required=True)
    alias = ui.ask('Embedding 服务商代号', 'Embedding provider alias', model.get('source', 'embedding'), required=True)
    model['source'] = alias
    model['url'] = ui.ask('完整 embedding 请求 URL', 'Complete embedding request URL', model.get('url'), required=True)
    from .agent.contracts import endpoint
    endpoint(str(model['url']))
    providers = draft.project.setdefault('providers', tomlkit.table())
    if alias not in providers or not default(providers[alias].get('url')):
        providers[alias] = dict(url=str(model['url']))
    secrets = draft.secrets.setdefault('providers', tomlkit.table()).setdefault(alias, tomlkit.table())
    secrets['key'] = ui.credential(alias, secrets.get('key', ''))
    model_advanced(ui, model, draft.project.get('pricing', {}).get(str(model['model'])), embedding=True)
    run = draft.project['run']
    ui.say(f"chunk_size={run.get('chunk_size', 2048)}, overlap_percent={run.get('overlap_percent', 20)}, text_extension={run.get('text_extension', 1)}")
    if ui.yes('修改检索参数？', 'Edit retrieval parameters?'):
        for field, zh in [('chunk_size', '切块字符数'), ('overlap_percent', '重叠百分比'), ('text_extension', '相邻上下文块数')]:
            run[field] = ui.number(zh, field, run[field], integer=True, required=True, positive=field == 'chunk_size')


def verification_group(ui: Console, draft: StudyDraft):
    models = draft.project['models']
    names = [str(models[k]['model']) for k in draft.project['selected_models']]
    run = draft.project['run']
    ui.say('模型互相核验，并排除模型自评：', 'Models cross-verify, excluding self-verification:')
    for name in names:
        ui.say(f"  {name} -> {', '.join(n for n in names if n != name)}")
    scores = [int(v) for v in ui.ask('核验票阈值（逗号分隔）', 'Verification thresholds (comma separated)',
                                    ','.join(str(v) for v in run.get('min_cross_scores', [1]))).split(',') if v.strip()]
    if not scores or len(set(scores)) != len(scores) or any(not 1 <= s < len(names) for s in scores):
        raise ValueError(f'Verification thresholds must be unique integers from 1 to {len(names)-1}')
    run['min_cross_scores'] = scores
    threshold = ui.number('共识模型数阈值', 'Distinct-model consensus threshold', run.get('min_consensus_models', len(names)//2+1), integer=True, required=True, positive=True)
    if threshold > len(names):
        raise ValueError('Consensus threshold exceeds selected models')
    run['min_consensus_models'] = threshold


def expected_quotes(draft: StudyDraft) -> dict:
    providers = draft.project['providers']
    selected = {str(draft.project['models'][k]['model']): str(providers[draft.project['models'][k]['source']]['url'])
                for k in draft.project['selected_models']}
    embedding = draft.project['embedding']
    selected[str(embedding['model'])] = str(embedding['url'])
    return selected


def costs_group(ui: Console, draft: StudyDraft):
    project, run = draft.project['project'], draft.project['run']
    rounds = [int(v.strip()) for v in ui.ask('运行轮次列表', 'Round indices', ','.join(str(v) for v in project.get('rounds', [1]))).split(',') if v.strip()]
    if not rounds or min(rounds) < 1 or len(set(rounds)) != len(rounds):
        raise ValueError('Rounds must be unique positive integers')
    project['rounds'] = rounds
    run['round_index'] = min(rounds)
    project['smoke_size'] = ui.number('试跑文章数', 'Trial paper count', project.get('smoke_size', 1), integer=True, required=True, positive=True)
    run['temperature'] = ui.number('温度', 'Temperature', run.get('temperature', .01), required=True)
    prices = draft.project.setdefault('pricing', tomlkit.table())
    expected = expected_quotes(draft)
    source = ui.ask('报价 JSON 路径或公开 URL（可跳过）', 'Quote JSON path or public URL (optional)')
    if source:
        try:
            ui.say('正在读取供应商报价…', 'Reading provider quotes...')
            ui.log.record('quote_fetch_started', source=source)
            matched, notes = read_quotes(source.strip('"'), expected)
            ui.say(json.dumps(redact(matched, ui.log.secrets), ensure_ascii=False, indent=2))
            for note in notes:
                ui.say(note)
            ui.log.record('quotes_previewed', source=source, matched=matched, diagnostics=notes)
            if matched and ui.yes('采用以上报价？', 'Apply these quotes?'):
                for model, quote in matched.items():
                    price = prices.setdefault(model, tomlkit.table())
                    price.update(quote)
        except (ValueError, OSError) as exc:
            ui.say(f'报价导入失败，可手填或保留未知：{exc}', f'Quote import failed; enter prices or leave unknown: {exc}')
            ui.log.record('quote_import_failed', source=source, error=str(exc))
    for model in expected:
        price = prices.setdefault(model, tomlkit.table())
        input_rate, output_rate = default(price.get('input_per_million')), default(price.get('output_per_million'))
        existing = f'{input_rate},{output_rate}' if input_rate is not None and output_rate is not None else ''
        raw = ui.ask(f'{model} 输入/输出单价（每百万 token，逗号分隔；- 清空）',
                     f'{model} input/output rates per million tokens (comma separated; - clears)', existing)
        if raw == '-' or not raw:
            price.pop('input_per_million', None)
            price.pop('output_per_million', None)
        else:
            values = [float(v.strip()) for v in raw.split(',')]
            if len(values) != 2 or any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('Provide two finite nonnegative rates')
            if raw != existing:
                for field in ('as_of', 'fetched_at', 'basis'):
                    price.pop(field, None)
                price['source'] = 'manual'
                price['base_url'] = expected[model]
            price['input_per_million'], price['output_per_million'] = values
            currency = ui.ask('币种（如 USD/CNY，可空）', 'Currency (USD/CNY, optional)', price.get('currency', ''))
            if currency.upper() != price.get('currency', ''):
                for field in ('as_of', 'fetched_at', 'basis'):
                    price.pop(field, None)
                price['source'] = 'manual'
                price['base_url'] = expected[model]
            optional(price, 'currency', currency.upper())
            price['basis'] = ui.ask('价格依据', 'Price basis', default(price.get('basis')) or 'User supplied', required=True)
    for model in list(prices):
        if model not in expected:
            del prices[model]
    budget = draft.project.setdefault('budget', tomlkit.table())
    ui.say('预算留空表示未设对应上限；未知价格不会按零计费。', 'Blank budgets mean no corresponding total cap; unknown prices are never zero.')
    for field, zh, en in [('max_calls', '总调用次数上限', 'Total call cap'), ('max_tokens', '总 token 上限', 'Total token cap'),
                           ('max_cost', '金额上限（配置币种）', 'Money cap (configured currency)'), ('max_runtime', '总运行时间上限（秒，含审核等待）', 'Wall-time cap (seconds, includes review)')]:
        optional(budget, field, ui.number(zh, en, default(budget.get(field)), integer=field in {'max_calls', 'max_tokens'}, positive=True))


def show_summary(ui: Console, draft: StudyDraft):
    ui.say('\n配置摘要', '\nConfiguration summary')
    ui.say(tomlkit.dumps(draft.project))
    ui.say(f'文献：{len(draft.papers)} 篇；问题：{len(draft.questions["questions"])} 个',
           f'Papers: {len(draft.papers)}; questions: {len(draft.questions["questions"])}')
    ui.say(f'保存位置：{draft.target}', f'Save destination: {draft.target}')
    if any(str(draft.project['project'].get(k, v)) != v for k, v in
           [('papers', 'papers.txt'), ('questions', 'questions.toml'), ('secrets', 'secrets.local.toml')]):
        ui.say('保存时会将已加载内容复制为本目录的四个标准配置文件，外部引用源不会被改写。',
               'Saving copies loaded content into four standard local files; external source files are not overwritten.')
    ui.say('API 密钥：[已设置/单独保存，不显示]', 'API keys: [stored separately; not displayed]')


def save_draft(ui: Console, draft: StudyDraft):
    path = draft.write(ui.log.directory / 'draft' / 'project.toml', draft=True)
    prefs = dict(language=ui.language, last_config=str(path), draft_target=str(draft.target))
    atomic_text(ROOT / '.lumina-ui.json', json.dumps(prefs, ensure_ascii=False), private=True)
    ui.log.record('draft_saved', path=str(path), target=str(draft.target))
    ui.say(f'草稿已保存：{path}', f'Draft saved: {path}')


def configure(ui: Console, draft: StudyDraft) -> int:
    groups = [('研究任务', 'Study', study_group), ('Stage 1：文献与抽取', 'Stage 1: papers and extraction', extraction_group),
              ('Stage 2：证据检索', 'Stage 2: evidence retrieval', embedding_group),
              ('Stage 3：核验与共识', 'Stage 3: verification and consensus', verification_group),
              ('运行与费用', 'Run and costs', costs_group)]
    index = 0
    try:
        while True:
            while index < len(groups):
                zh, en, function = groups[index]
                ui.say(f'\n[{index+1}/{len(groups)}] {zh}', f'\n[{index+1}/{len(groups)}] {en}')
                try:
                    function(ui, draft)
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    ui.say(f'设置需要修正：{exc}', f'Settings need correction: {exc}')
                    ui.log.record('group_invalid', group=en, error=str(exc))
                    action = ui.choice('1重填 / 2返回上组 / 3保存草稿退出', '1 retry / 2 back / 3 save draft and exit', {'1', '2', '3'})
                    if action == '3':
                        save_draft(ui, draft)
                        return 2
                    index = max(0, index - 1) if action == '2' else index
                    continue
                ui.log.record('group_completed', group=en)
                action = ui.choice('1继续 / 2返回上组 / 3保存草稿退出', '1 continue / 2 back / 3 save draft and exit', {'1', '2', '3'})
                if action == '3':
                    save_draft(ui, draft)
                    return 2
                index = max(0, index - 1) if action == '2' else index + 1
            draft.project['project']['language'] = ui.language
            show_summary(ui, draft)
            action = ui.choice('1只保存 / 2检查配置 / 3开始试跑 / 4返回修改 / 5保存草稿退出',
                               '1 save / 2 validate / 3 start trial / 4 edit / 5 save draft and exit', {'1', '2', '3', '4', '5'}, '2')
            if action == '5':
                save_draft(ui, draft)
                return 2
            if action == '4':
                index = 0
                continue
            try:
                ui.say('正在校验配置与输入文献…', 'Validating configuration and input papers...')
                spec = draft.validate()
            except (ValueError, OSError, TypeError, KeyError, SystemExit) as exc:
                ui.say(f'配置校验未通过：{exc}', f'Configuration validation failed: {exc}')
                ui.log.record('validation_failed', error=str(exc))
                index = int(ui.choice('返回设置组（1至5）', 'Return to setting group (1 to 5)', {'1', '2', '3', '4', '5'}, '5')) - 1
                continue
            ui.say('配置校验通过；未调用模型。', 'Configuration valid; no model calls made.')
            ui.log.record('validated', spec_hash=spec.fingerprint)
            if action == '2':
                continue
            if action == '3':
                project = draft.project['project']
                run_dir = (draft.base / str(project['runs_dir']) / str(project['run_id'])).resolve()
                if run_dir.exists():
                    ui.say('此运行标识已存在。修改配置需使用新的运行标识；续跑请返回启动菜单。',
                           'Run ID already exists. Edited settings require a new ID; use the start menu to resume.')
                    project['run_id'] = ui.ask('新的运行标识', 'New run ID', 'study-' + datetime.now().strftime('%Y%m%d-%H%M%S'), required=True)
                    continue
            if draft.target.exists() and not ui.yes('确认写回这四个配置文件？', 'Confirm writing the four configuration files?'):
                continue
            draft.write(draft.target)
            save_preferences(ROOT / '.lumina-ui.json', ui.language, draft.target)
            ui.log.record('configuration_saved', path=str(draft.target), project=draft.project.unwrap(), papers=draft.papers,
                          questions=draft.questions.unwrap())
            ui.say(f'配置已保存：{draft.target}', f'Configuration saved: {draft.target}')
            if action == '1':
                return 0
            if not ui.yes('开始付费模型试跑？', 'Start the trial with model API calls?'):
                return 0
            from .study import run_study
            return run_study(str(draft.target), load_config(str(draft.target)), setup_log=ui.log.directory)
    except (EOFError, KeyboardInterrupt):
        ui.say('\n输入已中断，将保存草稿。', '\nInput interrupted; saving a draft.')
        save_draft(ui, draft)
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='LUMINA interactive study setup')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--lang', choices=['zh', 'en'])
    parser.add_argument('--setup-log', type=Path)
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        print('Interactive setup requires a terminal. Use run_pipeline.py --config PATH --check for automation.', file=sys.stderr)
        return 2
    folder = args.setup_log or ROOT / 'runs' / '_setup' / (datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8])
    log = SetupLog(folder.resolve())
    preferences = read_preferences(ROOT / '.lumina-ui.json')
    language = args.lang or preferences.get('language', 'zh')
    language = language if language in {'zh', 'en'} else 'zh'
    ui = Console(language, log)
    draft = None
    try:
        print(BANNER)
        if not args.lang:
            language = ui.choice('语言 / Language: 1中文 / 2English', 'Language / 语言: 1中文 / 2English', {'1', '2'}, '1' if language == 'zh' else '2')
            ui.language = 'zh' if language == '1' else 'en'
        log.record('welcome', language=ui.language, interpreter=sys.executable)
        action = ui.choice('1新建研究任务 / 2加载配置并修改 / 3继续已有运行',
                           '1 new study / 2 load and edit / 3 resume existing run', {'1', '2', '3'}, '2' if args.config else '1')
        if action == '1':
            target = Path(ui.ask('配置目录', 'Configuration folder', 'configs/my-study', required=True).strip('"')).expanduser().resolve()
            if target.exists() and any(target.iterdir()):
                raise ValueError('Study folder is not empty; choose load/edit or a new folder')
            template = ui.choice('模板：aqua / wildfire / generic', 'Template: aqua / wildfire / generic', {'aqua', 'wildfire', 'generic'}, 'generic')
            draft = StudyDraft.create(target / 'project.toml', template)
        else:
            selected = args.config or preferences.get('last_config')
            path = Path(ui.ask('project.toml 路径', 'project.toml path', str(selected) if selected else None, required=True).strip('"')).expanduser().resolve()
            if action == '3':
                config = load_config(str(path))
                for values in config.LLM_SETTINGS.values():
                    log.protect(values.get('key', ''))
                run_path = Path(config.PROJECT['runs_dir']) / config.PROJECT['run_id']
                if not run_path.is_dir():
                    raise ValueError('No existing run matches this configuration; choose new/edit')
                config.PROJECT['language'] = ui.language
                log.record('resume_requested', config=str(path), run=str(run_path))
                save_preferences(ROOT / '.lumina-ui.json', ui.language, path)
                from .study import run_study
                return run_study(str(path), config, setup_log=log.directory)
            draft = StudyDraft.load(path)
            if str(path) == preferences.get('last_config') and preferences.get('draft_target'):
                draft.target = Path(preferences['draft_target']).resolve()
                draft.base = draft.target.parent
            for values in draft.secrets.get('providers', {}).values():
                log.protect(str(values.get('key', '')))
            log.record('configuration_loaded', path=str(path))
        return configure(ui, draft)
    except (EOFError, KeyboardInterrupt):
        log.record('setup_cancelled')
        return 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.record('setup_failed', error=str(exc))
        ui.say(f'启动失败：{exc}', f'Setup failed: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
