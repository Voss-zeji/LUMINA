from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lumina import prompts
from lumina.common import validate_answers
from lumina.configuration import load_config
from lumina.study import run_study

ROOT = Path(__file__).resolve().parents[1]


def study_files(root: Path, domain='generic') -> Path:
    folder = root / '研究 配置'
    shutil.copytree(ROOT / 'configs' / domain, folder)
    path = folder / 'project.toml'
    text = path.read_text(encoding='utf-8')
    for a, b in [('REPLACE-model-a', 'mock/Model_A'), ('REPLACE-model-b', 'mock/Model_B'),
                 ('REPLACE-embedding', 'mock/embedding'), ('REPLACE.invalid', 'example.invalid'),
                 ('../../runs', '../runs'), ('rate_limit_seconds = 1', 'rate_limit_seconds = 0')]:
        text = text.replace(a, b)
    for field, value in dict(max_calls=200, max_tokens=2000000, max_cost=5, max_runtime=600,
                             input_per_million=1, output_per_million=2,
                             max_input_tokens=20000, max_output_tokens=1000).items():
        text = text.replace(f'{field} = "REPLACE"', f'{field} = {value}')
    text = text.replace('REPLACE: provider, date and price source', 'synthetic fixture; not a real price')
    path.write_text(text, encoding='utf-8')
    paper = folder / '文章 01.md'
    paper.write_text('Habitat is forest. Biomass is 2 kg.', encoding='utf-8')
    (folder / 'papers.txt').write_text('# Explicit papers\n文章 01.md\n', encoding='utf-8')
    (folder / 'secrets.local.toml').write_text('[providers.chat]\nkey="SECRET-DO-NOT-PERSIST"\n'
                                             '[providers.embedding]\nkey="SECRET-DO-NOT-PERSIST"\n', encoding='utf-8')
    return path


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = study_files(self.root)

    def test_relative_unicode_inputs_isolation_and_no_network_check(self):
        cfg = load_config(str(self.path))
        self.assertTrue(Path(cfg.REQUEST['papers'][0]).is_file())
        self.assertEqual(Path(cfg.PROJECT['runs_dir']), (self.root / 'runs').resolve())
        first = cfg.DOMAINS['generic']['question_set']
        first['questions'][0]['prompt'] = 'changed in memory'
        self.assertNotEqual(first, load_config(str(self.path)).DOMAINS['generic']['question_set'])
        with patch('lumina.agent.runtime.OpenAI') as sdk, patch('lumina.agent.runtime.requests.post') as post, redirect_stdout(io.StringIO()):
            self.assertEqual(run_study(str(self.path), cfg, check=True), 0)
        sdk.assert_not_called()
        post.assert_not_called()
        self.assertFalse((self.root / 'runs').exists())

    def test_invalid_configuration_rejected_before_requests(self):
        original = self.path.read_text(encoding='utf-8')
        cases = [('run_id = "my-study-001"', 'run_id = "../escape"'),
                 ('domain = "generic"', 'domain = "../escape"'),
                 ('stage = "all"', 'stage = "unknown"'),
                 ('min_cross_scores = [1]', 'min_cross_scores = [2]'),
                 ('max_cost = 5', 'max_cost = -1'),
                 ('timeout_seconds = 120', 'timeout_seconds = 0'),
                 ('selected_models = ["model_a", "model_b"]', 'selected_models = ["model_a", "model_a"]'),
                 ('[budget]', '[unknown_budget]'),
                 ('supports_json_mode = true', 'supports_json_mode = true\nkey="bad-location"')]
        for before, after in cases:
            with self.subTest(after=after), patch('lumina.agent.runtime.OpenAI') as api:
                self.path.write_text(original.replace(before, after), encoding='utf-8')
                with self.assertRaises((ValueError, OSError)):
                    load_config(str(self.path))
                api.assert_not_called()
        self.path.write_text(original, encoding='utf-8')
        (self.path.parent / 'papers.txt').write_text('missing.md\n', encoding='utf-8')
        with self.assertRaises(FileNotFoundError):
            load_config(str(self.path))

    def test_question_validation_and_reversed_field_types(self):
        definition = prompts.definition('aqua')
        for field, value in [('kind', 'nested'), ('id', definition['questions'][1]['id']), ('require_unit', 'true')]:
            changed = copy.deepcopy(definition)
            changed['questions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prompts.validate_definition(changed)
        definition['templates']['instruction'] += '{missing}'
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            prompts.validate_definition(definition)
        cfg = load_config(str(self.path)).DOMAINS['generic']
        row = dict(item='Biomass', value=2, evidence='measured', confidence_lv=95, unit='kg', experimental='plot A')
        validate_answers(pd.DataFrame([row]), 'generic', 2, cfg)
        for field in ('unit', 'experimental'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_answers(pd.DataFrame([{k: v for k, v in row.items() if k != field}]), 'generic', 2, cfg)
        with self.assertRaisesRegex(ValueError, 'experimental'):
            validate_answers(pd.DataFrame([row, dict(row, experimental=None)]), 'generic', 2, cfg)
        validate_answers(pd.DataFrame([row, dict(row, value=None, unit=None, experimental=None)]), 'generic', 2, cfg)
        cfg['question_set']['questions'][0]['allowed_values'] = ['forest']
        with self.assertRaises(ValueError):
            validate_answers(pd.DataFrame([dict(row, item='Habitat', value='ocean')]), 'generic', 1, cfg)

    def test_builtin_prompt_bytes_match_f9036f7(self):
        hashes = {
            'message_system_v2': 'ecbf8987ce66dd32c93d73ea5a752f58286c7f07a6bb6933dda32e6dcc3f9824',
            'message_system_v2_output': 'acc323a248c273a9aa9a7cc42597d8fe3968785d954f4b927c4deac02eb1f9ff',
            'message_system_ragQuery': 'd84543d40ca9b427934a2fdaa87a4a0125b9e892c5ea69a06834fd41865d2eb4',
            'checker_requery': '1aa1493e40c86477ab653fb35300044c413c98b3c568539256d96382134bf8c5',
            'checker_requeryFull': '9ac6b2e4119c449ef110ff69bdb4f55f30a48871b7e0e0641d15cec30b13b360',
            'aqua_question_A_meta': '7eb4a861df4ced269c0e0fda7ba42c30cda9ff2ffabc8072f05e468fd2a05528',
            'aqua_question_B_experiment': 'e4ecee52a606d946423a530e2eae6a8561d9d6d711fe2a5d72512f6e98951f81',
            'aqua_question_C_flux': 'b8401a9b65e3eb6676b2f7a24aa344b57369e79977dbffe3b13659ad211cbd44',
            'wildfire_question_A_meta': '06a2d2017f44cd2590f5a1cc6e810deb299b6304ea0615d8483df8949850c4f4',
            'wildfire_question_B_ef_details': '9ccc1be08c2fc3becedaa4d95f53fadca495b5a2135bbac4d4e4c96cd2f61cb2',
        }
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256(getattr(prompts, name).encode()).hexdigest(), digest, name)

    def test_custom_templates_do_not_read_builtin_files(self):
        from lumina.cross_validation import verifier_signatures
        cfg = load_config(str(self.path))
        with patch.object(prompts, '_builtin', side_effect=AssertionError('Custom study read built-in prompts')):
            signatures = verifier_signatures(cfg.FULL_LLM_POOL, cfg.LLM_SETTINGS, cfg.EMBEDDING_MODEL,
                                              cfg.RUN, cfg.DOMAINS['generic'])
        self.assertEqual(set(signatures), {'Model_A', 'Model_B'})

    def test_generic_toml_through_real_worker_then_resume_is_offline(self):
        # Test-only transport injection in child interpreters; production has no mock switch.
        hooks = self.root / 'hooks'
        hooks.mkdir()
        mock = '''
from types import SimpleNamespace
from unittest.mock import Mock
from lumina.agent import runtime
import json
def reply(**payload):
    question = payload['messages'][-1]['content']
    if question.startswith('Return the habitat'):
        content = {'Habitat': dict(value='forest', evidence='Habitat is forest.', confidence_lv=95)}
    elif question.startswith('Extract biomass'):
        content = {'Biomass': dict(value=2, unit='kg', experimental='plot A', evidence='Biomass is 2 kg.', confidence_lv=95)}
    else:
        content = dict(existing_flag=1, direct_quote='Habitat is forest. Biomass is 2 kg.')
    response = dict(choices=[dict(message=dict(content=json.dumps(content)),finish_reason='stop')],usage=dict(prompt_tokens=12,completion_tokens=5,total_tokens=17))
    return SimpleNamespace(model_dump=lambda **_: response)
client=Mock()
client.chat.completions.create.side_effect=reply
runtime.OpenAI=lambda **_:client
embedding=Mock()
embedding.json.return_value=dict(data=[dict(embedding=[1.,0.])],usage=dict(prompt_tokens=3,total_tokens=3))
runtime.requests.post=lambda *a,**k:embedding
runtime.RunContext.sleep=lambda *a,**k:None
'''
        (hooks / 'sitecustomize.py').write_text((ROOT / 'tests' / 'sitecustomize.py').read_text() + mock, encoding='utf-8')
        env = dict(os.environ, LUMINA_TEST_OFFLINE='1', PYTHONIOENCODING='utf-8', PYTHONPATH=os.pathsep.join([str(hooks), str(ROOT)]))

        def cli(args, expected):
            child = subprocess.run([sys.executable, str(ROOT / 'run_pipeline.py'), *args], cwd=self.root,
                                   env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding='utf-8', timeout=45)
            self.assertEqual(child.returncode, expected, child.stdout + child.stderr)
            self.assertNotIn('SECRET-DO-NOT-PERSIST', child.stdout + child.stderr)
            return child.stdout

        args = ['--config', str(self.path)]
        cli([*args, '--check'], 0)
        self.assertFalse((self.root / 'runs').exists())
        cli(args, 2)
        run = self.root / 'runs' / 'my-study-001'
        from lumina.agent.store import Store
        with Store(run) as store:
            gate = next(g for g in store.gates() if g['kind'] == 'smoke')
        cli(['approve', '--runs-dir', str(run.parent), '--run-id', run.name, '--gate-id', gate['gate_id'], '--reason', 'Reviewed synthetic trial'], 0)
        cli(args, 0)
        with Store(run) as store:
            calls = store._conn.execute('SELECT count(*) FROM attempts').fetchone()[0]
        cli(args, 0)
        with Store(run) as store:
            self.assertEqual(store._conn.execute('SELECT count(*) FROM attempts').fetchone()[0], calls)
        report = json.loads((run / 'reports' / 'final_report.json').read_text(encoding='utf-8'))
        self.assertTrue(report['final_qc']['passed'])
        self.assertNotIn('SECRET-DO-NOT-PERSIST', (run / 'manifest.json').read_text(encoding='utf-8'))
