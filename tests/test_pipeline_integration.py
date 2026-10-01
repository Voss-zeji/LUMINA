from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import run_pipeline
from lumina import common, cross_validation as cv, examiner, llm, preparation, prompts


class PipelineIntegration(unittest.TestCase):
    def config(self, root, domain):
        cfg = {k: str(root / k) for k in ('markdown_dir', 'pdf_dir', 'examiner_output', 'composite_dir',
                                        'embedding_dir', 'crosser_dir', 'ensemble_dir')}
        Path(cfg['markdown_dir']).mkdir()
        (Path(cfg['markdown_dir']) / '001_test.md').write_text('China. 2020. Fish. Table 2. Flux is 2.', encoding='utf-8')
        cfg.update(domain_knowledge='test', composite_prefix='C', questions=list(range(1, 4 if domain == 'aqua' else 5)))
        models = {'a': dict(model='Model_A', source='mock'), 'b': dict(model='Model_B', source='mock')}
        return SimpleNamespace(DOMAINS={domain: cfg}, FULL_LLM_POOL=models, SELECTED_KEYS=['a', 'b'],
            LLM_SETTINGS={'mock': {}}, EMBEDDING_MODEL={}, RUN=dict(round_index=1, temperature=.01,
            chunk_size=2048, overlap_percent=20, text_extension=1, min_cross_scores=[1]))

    def response(self, domain):
        def chat(_model, _settings, messages, **_kwargs):
            if messages[0]['content'] == prompts.message_system_ragQuery.strip():
                return json.dumps(dict(existing_flag=1, direct_quote='China')), None
            q = [x.strip('\n') for x in prompts.questions_for_domain(domain)].index(messages[-1]['content']) + 1
            if q == 1:
                item, value = 'Study_location', 'China'
            elif domain == 'aqua' and q == 2:
                item, value = 'Specie', 'fish'
            else:
                item, value = ('Flux-1' if domain == 'aqua' else 'Forest_smoldering'), 2
            answer = dict(value=value, evidence='China', confidence_lv=95)
            if domain == 'aqua' and q == 3:
                answer['unit'] = 'mg m-2 h-1'
            return json.dumps({item: answer}), None
        return chat

    def test_both_domains_all_then_resume_and_cross_again(self):
        for domain, count in [('aqua', 12), ('wildfire', 16)]:
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as temp:
                config = self.config(Path(temp), domain)
                with patch.object(examiner, 'single_chat', side_effect=self.response(domain)) as extract, \
                     patch.object(llm, 'single_chat', side_effect=self.response(domain)) as verify, \
                     patch.object(cv, 'embedding_response', return_value=[1., 0.]) as embed, \
                     patch.object(preparation, 'token_calculator', side_effect=lambda s: len(s.split())), \
                     patch.object(examiner, 'sleep_for_rate_limit'), patch.object(cv, 'sleep_for_rate_limit'), \
                     patch.object(llm.requests, 'post', side_effect=AssertionError('unexpected real network')):
                    run_pipeline.run(domain, 'all', config)
                    first_counts = (extract.call_count, verify.call_count, embed.call_count)
                    run_pipeline.run(domain, 'all', config)
                    self.assertEqual((extract.call_count, verify.call_count, embed.call_count), first_counts)
                    run_pipeline.run(domain, 'cross', config)
                    self.assertEqual((extract.call_count, verify.call_count, embed.call_count), first_counts)
                cfg = config.DOMAINS[domain]
                self.assertEqual(len(list(Path(cfg['ensemble_dir']).rglob('*.xlsx'))), count)
                df = pd.read_excel(Path(cfg['composite_dir']) / 'C_Q01.xlsx', dtype={'paper_index': str})
                self.assertEqual(set(df.paper_index), {'001'})
                self.assertEqual(set(df.cross_score), {1})
                status = json.loads((Path(cfg['ensemble_dir']) / 'last_run.json').read_text(encoding='utf-8'))
                self.assertEqual(status['status'], 'succeeded')
                scores = pd.read_excel(Path(cfg['crosser_dir']) / 'cross_scores.xlsx')
                self.assertIn('cross_score', scores)

    def test_stale_round_or_temperature_rejected_before_cross_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), 'aqua')
            with patch.object(examiner, 'single_chat', side_effect=self.response('aqua')), patch.object(examiner, 'sleep_for_rate_limit'):
                run_pipeline.run('aqua', 'examiner', config)
            run_pipeline.run('aqua', 'composite', config)
            for key, value in [('round_index', 2), ('temperature', .5)]:
                original = config.RUN[key]; config.RUN[key] = value
                with patch.object(cv, 'embedding_response') as api:
                    with self.assertRaisesRegex(ValueError, 'stale composite'):
                        run_pipeline.run('aqua', 'cross', config)
                api.assert_not_called()
                config.RUN[key] = original

    def test_recoverable_sidecar_keeps_canonical_model_with_underscore_and_case(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), 'aqua'); cfg = config.DOMAINS['aqua']
            with patch.object(examiner, 'single_chat', side_effect=self.response('aqua')), patch.object(examiner, 'sleep_for_rate_limit'):
                run_pipeline.run('aqua', 'examiner', config)
            tasks = examiner.expected_tasks('aqua', cfg, config.FULL_LLM_POOL, config.LLM_SETTINGS, config.RUN)
            output = next(iter(tasks)); Path(output).unlink()
            common.invalid_path(output).write_text(json.dumps({'Study_location':dict(value='China', evidence='China', confidence_lv=95)}), encoding='utf-8')
            run_pipeline.run('aqua', 'composite', config)
            df = pd.read_excel(Path(cfg['composite_dir']) / 'C_Q01.xlsx')
            self.assertEqual(set(df.model), {'Model_A', 'Model_B'})
            self.assertEqual(len(df), 2)

    def test_changed_content_and_embedding_model_rebuild_equal_shape_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), 'aqua'); cfg = config.DOMAINS['aqua']
            with patch.object(cv, 'embedding_response', return_value=[1., 0.]) as api:
                cv.generate_embeddings_for_domain(cfg, {}, {}, config.RUN)
                self.assertEqual(api.call_count, 1)
                cv.generate_embeddings_for_domain(cfg, {}, {}, config.RUN)
                self.assertEqual(api.call_count, 1)
                (Path(cfg['markdown_dir']) / '001_test.md').write_text('Changed short text.', encoding='utf-8')
                cv.generate_embeddings_for_domain(cfg, {}, {}, config.RUN)
                self.assertEqual(api.call_count, 2)
                cv.generate_embeddings_for_domain(cfg, {'model':'new-model'}, {}, config.RUN)
                self.assertEqual(api.call_count, 3)

    def test_atomic_failure_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.csv'; path.write_text('old', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                with common.atomic_output(path) as staged:
                    staged.write_text('partial', encoding='utf-8')
                    raise RuntimeError('interrupted')
            self.assertEqual(path.read_text(encoding='utf-8'), 'old')
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_literal_na_paper_id_is_preserved_as_an_identifier(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), 'aqua'); cfg = config.DOMAINS['aqua']
            path = Path(cfg['markdown_dir']) / '001_test.md'; path.rename(path.with_name('NA.md'))
            with patch.object(examiner, 'single_chat', side_effect=self.response('aqua')), \
                 patch.object(llm, 'single_chat', side_effect=self.response('aqua')), \
                 patch.object(cv, 'embedding_response', return_value=[1., 0.]), \
                 patch.object(preparation, 'token_calculator', side_effect=lambda s: len(s.split())), \
                 patch.object(examiner, 'sleep_for_rate_limit'), patch.object(cv, 'sleep_for_rate_limit'):
                run_pipeline.run('aqua', 'all', config)
                run_pipeline.run('aqua', 'all', config)
            frame = common.read_composite(Path(cfg['composite_dir']) / 'C_Q01.xlsx')
            self.assertEqual(set(frame.paper_index), {'NA'})
            self.assertEqual(set(frame.cross_score), {1})

    def test_cli_lock_excludes_a_second_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'run.lock'
            with common.pipeline_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'lock exists'):
                    with common.pipeline_lock(path):
                        self.fail('second writer entered')
            self.assertFalse(path.exists())

    def test_absent_or_changed_cross_proof_cannot_complete_filtered_ensemble(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), 'aqua'); cfg = config.DOMAINS['aqua']
            with patch.object(examiner, 'single_chat', side_effect=self.response('aqua')), \
                 patch.object(llm, 'single_chat', side_effect=self.response('aqua')), \
                 patch.object(cv, 'embedding_response', return_value=[1., 0.]), \
                 patch.object(preparation, 'token_calculator', side_effect=lambda s: len(s.split())), \
                 patch.object(examiner, 'sleep_for_rate_limit'), patch.object(cv, 'sleep_for_rate_limit'):
                run_pipeline.run('aqua', 'all', config)
            config.EMBEDDING_MODEL = {'model':'changed-embedding-model'}
            with self.assertRaisesRegex(ValueError, 'cross coverage'):
                run_pipeline.run('aqua', 'ensemble', config)
            status = json.loads((Path(cfg['ensemble_dir']) / 'last_run.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'failed')

    def test_no_evidence_missing_answers_remain_a_valid_empty_filtered_case(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), 'aqua'); cfg = config.DOMAINS['aqua']
            normal = self.response('aqua')
            def missing(*args, **kwargs):
                content, tokens = normal(*args, **kwargs)
                parsed = json.loads(content)
                for item in parsed.values():
                    item.update(value=None, evidence=None)
                    if 'unit' in item:
                        item['unit'] = None
                return json.dumps(parsed), tokens
            with patch.object(examiner, 'single_chat', side_effect=missing), \
                 patch.object(llm, 'single_chat', side_effect=AssertionError('no evidence to verify')), \
                 patch.object(cv, 'embedding_response', return_value=[1., 0.]), \
                 patch.object(preparation, 'token_calculator', side_effect=lambda s: len(s.split())), \
                 patch.object(examiner, 'sleep_for_rate_limit'), patch.object(cv, 'sleep_for_rate_limit'):
                run_pipeline.run('aqua', 'all', config)
            result = pd.read_excel(Path(cfg['ensemble_dir']) / 'MiniCross01/aqua_Ensemble_Result_Q03_MiniCross01.xlsx')
            self.assertTrue(result.empty)
            self.assertIn('unit', result)

    def test_partial_verifier_failure_is_not_accepted_as_complete_ensemble(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.config(Path(temp), 'aqua')
            config.FULL_LLM_POOL['c'] = dict(model='Model_C', source='mock'); config.SELECTED_KEYS.append('c')
            normal = self.response('aqua')
            def verifier(model, *args, **kwargs):
                if model['model'] == 'Model_C':
                    raise RuntimeError('unavailable verifier')
                return normal(model, *args, **kwargs)
            with patch.object(examiner, 'single_chat', side_effect=normal), patch.object(llm, 'single_chat', side_effect=verifier), \
                 patch.object(cv, 'embedding_response', return_value=[1., 0.]), \
                 patch.object(preparation, 'token_calculator', side_effect=lambda s: len(s.split())), \
                 patch.object(examiner, 'sleep_for_rate_limit'), patch.object(cv, 'sleep_for_rate_limit'):
                with self.assertRaisesRegex(RuntimeError, 'cross failed'):
                    run_pipeline.run('aqua', 'all', config)
            with self.assertRaisesRegex(ValueError, 'cross coverage'):
                run_pipeline.run('aqua', 'ensemble', config)


if __name__ == '__main__':
    unittest.main()
