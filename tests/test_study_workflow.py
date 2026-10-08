from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from lumina import prompts
from lumina.agent.contracts import ResearchSpecification
from lumina.agent.controller import approve, begin, execute
from lumina.agent.store import Store
from lumina.study import run_study
from test_agent_contracts import config_for, request_for


class StudyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = config_for('ecology')
        definition = prompts.definition('aqua')
        numeric = copy.deepcopy(definition['questions'][2])
        numeric.update(id='mass', prompt='Return the measured biomass.', items=[], require_unit=True)
        text = copy.deepcopy(definition['questions'][0])
        text.update(id='habitat', prompt='Return the habitat.', items=['Habitat'], allowed_values=['forest'])
        definition['questions'] = [numeric, text]
        self.config.DOMAINS['ecology'].update(question_set=definition, questions=[1, 2])
        self.papers = []
        for index in range(2):
            path = self.root / f'文章 {index}.md'
            path.write_text(f'# Results\nHabitat is forest. Biomass is 2 kg. Sample {index}.', encoding='utf-8')
            self.papers.append(str(path))
        self.request = request_for(self.papers[0], 'ecology')
        self.request['papers'] = self.papers
        self.config.REQUEST = self.request
        self.config.PROJECT = dict(run_id='study', runs_dir=str(self.root / 'runs'), stage='all')
        self.client = Mock()
        self.client.chat.completions.create.side_effect = self.response
        embedding = Mock()
        embedding.json.return_value = dict(data=[dict(embedding=[1., 0.])], usage=dict(prompt_tokens=3, total_tokens=3))
        for patcher in [patch('lumina.agent.runtime.OpenAI', return_value=self.client),
                        patch('lumina.agent.runtime.requests.post', return_value=embedding),
                        patch('lumina.agent.runtime.RunContext.sleep')]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def response(self, **payload):
        message = payload['messages'][-1]['content']
        if message == 'Return the measured biomass.':
            content = {'Biomass': dict(value=2, unit='kg', evidence='Biomass is 2 kg.', confidence_lv=95)}
        elif message == 'Return the habitat.':
            content = {'Habitat': dict(value='forest', evidence='Habitat is forest.', confidence_lv=95)}
        else:
            content = dict(existing_flag=1, direct_quote='Biomass is 2 kg. Habitat is forest.')
        response = dict(choices=[dict(message=dict(content=json.dumps(content)), finish_reason='stop')],
                        usage=dict(prompt_tokens=12, completion_tokens=5, total_tokens=17))
        return SimpleNamespace(model_dump=lambda **_: response)

    def start(self):
        return begin(self.request, self.config, self.root / 'runs', run_id='study', dry_run=True)

    def test_new_topic_all_stages_trial_approval_and_cached_resume(self):
        run = self.start()
        first = execute(run, self.config)
        self.assertEqual(first['state'], 'HUMAN_GATE_SMOKE', first)
        trial_calls = self.client.chat.completions.create.call_count
        self.assertEqual(execute(run, self.config)['state'], 'HUMAN_GATE_SMOKE')
        self.assertEqual(self.client.chat.completions.create.call_count, trial_calls)
        with Store(run) as store:
            gate = next(g for g in store.gates() if g['kind'] == 'smoke')
        approve(run, gate['gate_id'], 'Reviewed synthetic trial')
        final = execute(run, self.config)
        self.assertEqual(final['state'], 'DONE', final)
        self.assertEqual(self.client.chat.completions.create.call_count, trial_calls * 2)
        files = list((run / 'outputs' / 'R01' / 'ensemble').rglob('*.xlsx'))
        self.assertEqual(len(files), 8)
        accepted = [pd.read_excel(p) for p in files if 'MiniCross' in str(p) and 'all' not in p.name.lower()]
        self.assertTrue(any('Habitat' in set(f.get('item', [])) for f in accepted))
        self.assertTrue(any('Biomass' in set(f.get('item', [])) for f in accepted))
        final_calls = self.client.chat.completions.create.call_count
        self.assertEqual(execute(run, self.config)['state'], 'DONE')
        self.assertEqual(self.client.chat.completions.create.call_count, final_calls)
        report = json.loads((run / 'reports' / 'final_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['extraction']['completed'], 8)
        self.assertNotIn('SECRET-DO-NOT-PERSIST', (run / 'manifest.json').read_text(encoding='utf-8'))
        from lumina.evaluation import evaluate_run
        evaluation = evaluate_run(run, output_dir=str(self.root / 'evaluation'))
        self.assertEqual(evaluation['status'], 'NOT_EVALUATED')
        self.assertTrue(evaluation['production_unchanged'])

    def test_prompt_rule_and_verification_changes_invalidate_identity(self):
        original = ResearchSpecification.from_config(self.request, self.config)
        for change in ('prompt', 'kind', 'items', 'verifier'):
            cfg = copy.deepcopy(self.config)
            definition = cfg.DOMAINS['ecology']['question_set']
            if change == 'verifier':
                definition['templates']['verifier_system'] += '\nBe careful.'
            elif change == 'items':
                definition['questions'][1]['items'] = ['Vegetation']
            elif change == 'kind':
                definition['questions'][0]['kind'] = 'text'
                definition['questions'][0]['require_unit'] = False
            else:
                definition['questions'][0]['prompt'] += ' Please.'
            with self.subTest(change=change):
                self.assertNotEqual(original.scientific_hash, ResearchSpecification.from_config(self.request, cfg).scientific_hash)

    def test_check_no_run_directory_or_request_and_noninteractive_trial_stops(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run_study('unused.toml', self.config, check=True), 0)
        self.assertFalse((self.root / 'runs').exists())
        self.client.chat.completions.create.assert_not_called()
        with patch('lumina.study.supervise', side_effect=lambda run, *_a, **_k: execute(run, self.config)), \
             patch('lumina.study.sys.stdin.isatty', return_value=False), patch('builtins.input') as ask, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run_study('unused.toml', self.config), 2)
        ask.assert_not_called()
        with Store(self.root / 'runs' / 'study') as store:
            self.assertEqual(store.snapshot()['run']['state'], 'HUMAN_GATE_SMOKE')
            self.assertTrue(all(g['status'] == 'pending' for g in store.gates()))

    def test_stage_pause_then_continue_reuses_completed_paid_work(self):
        self.config.PROJECT['stage'] = 'examiner'
        run = self.start()
        self.assertEqual(execute(run, self.config)['state'], 'PAUSED')
        self.assertEqual(self.client.chat.completions.create.call_count, 4)
        self.config.PROJECT['stage'] = 'all'
        self.assertEqual(execute(run, self.config)['state'], 'HUMAN_GATE_SMOKE')
        self.assertEqual(self.client.chat.completions.create.call_count, 8)

    def test_interactive_approval_and_config_change_refusal(self):
        with patch('lumina.study.supervise', side_effect=lambda run, *_a, **_k: execute(run, self.config)), \
             patch('lumina.study.sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='yes'), \
             patch('lumina.configuration.load_config', return_value=self.config), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run_study('unused.toml', self.config), 0)
            calls = self.client.chat.completions.create.call_count
            self.assertEqual(run_study('unused.toml', self.config), 0)
            self.assertEqual(self.client.chat.completions.create.call_count, calls)
            self.config.DOMAINS['ecology']['question_set']['questions'][0]['prompt'] += ' Changed.'
            self.assertEqual(run_study('unused.toml', self.config), 1)
            self.assertEqual(self.client.chat.completions.create.call_count, calls)

    def test_configuration_changed_during_review_never_approves_batch(self):
        changed = copy.deepcopy(self.config)
        changed.DOMAINS['ecology']['question_set']['questions'][0]['prompt'] += ' New request.'
        with patch('lumina.study.supervise', side_effect=lambda run, *_a, **_k: execute(run, self.config)), \
             patch('lumina.study.sys.stdin.isatty', return_value=True), patch('builtins.input', return_value='yes'), \
             patch('lumina.configuration.load_config', return_value=changed), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run_study('unused.toml', self.config), 1)
        with Store(self.root / 'runs' / 'study') as store:
            self.assertTrue(all(g['status'] == 'pending' for g in store.gates()))

    def test_eof_during_trial_confirmation_is_a_safe_pause(self):
        with patch('lumina.study.supervise', side_effect=lambda run, *_a, **_k: execute(run, self.config)), \
             patch('lumina.study.sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=EOFError), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(run_study('unused.toml', self.config), 2)
