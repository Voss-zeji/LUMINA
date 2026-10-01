from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import run_pipeline
from lumina import common, composite, cross_validation as cv, ensemble, examiner, prompts


class ContractRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {k: str(self.root / k) for k in (
            'markdown_dir', 'pdf_dir', 'examiner_output', 'composite_dir',
            'embedding_dir', 'crosser_dir', 'ensemble_dir')}
        for path in self.cfg.values():
            Path(path).mkdir()
        self.cfg.update(composite_prefix='C', domain_knowledge='test', questions=[1, 2, 3])
        self.md = Path(self.cfg['markdown_dir']) / '01_paper.md'
        self.md.write_text('China. Fish in 2020. Methane flux is 2.', encoding='utf-8')
        self.models = {'a': {'model': 'model-a', 'source': 'mock'},
                       'b': {'model': 'model-b', 'source': 'mock'}}
        self.names = ['model-a', 'model-b']
        self.run = dict(round_index=1, temperature=.01, chunk_size=2048,
                        overlap_percent=20, text_extension=1, min_cross_scores=[1])

    def response(self, _llm, _settings, messages, **_kwargs):
        question = messages[-1]['content']
        qs = prompts.questions_for_domain('aqua')
        item, value = ('Study_location', 'China') if question == qs[0].strip('\n') else (
            ('Specie', 'fish') if question == qs[1].strip('\n') else ('Flux-1', 2))
        answer = dict(value=value, evidence='China', confidence_lv=90)
        if item == 'Flux-1':
            answer['unit'] = 'mg m-2 h-1'
        return json.dumps({item: answer}), 10

    def extract(self):
        with patch.object(examiner, 'single_chat', side_effect=self.response), patch.object(examiner, 'sleep_for_rate_limit'):
            examiner.run_examiner_for_domain('aqua', self.cfg, self.models, {'mock': {}}, self.run)

    def compose(self):
        return composite.create_baseline_composite(self.cfg, [1, 2, 3], models=self.names,
                                                  round_index=self.run['round_index'], domain='aqua',
                                                  expected_tasks=examiner.expected_tasks('aqua', self.cfg, self.models, {'mock': {}}, self.run))

    def cross(self, flag=1):
        with patch.object(cv, 'embedding_response', return_value=[1., 0.]) as embeds, patch.object(cv, 'llm_requery', return_value=(json.dumps(dict(existing_flag=flag, direct_quote='China')), 1)) as queries, patch.object(cv, 'sleep_for_rate_limit'):
            cv.cross_validate_domain('aqua', self.cfg, self.models, {'mock': {}}, {}, self.run, self.names)
        return embeds.call_count, queries.call_count

    def aggregate(self):
        return cv.aggregate_cross_scores(self.cfg, self.names,
            expected_verifiers=cv.verifier_signatures(self.models, {'mock': {}}, {}, self.run))

    def test_author_year_stems_have_distinct_ids(self):
        self.assertEqual(common.paper_prefix_from_path('Smith_2020.md'), 'Smith_2020')
        self.assertEqual(common.paper_prefix_from_path('Smith_2021.md'), 'Smith_2021')

    def test_toc_reference_does_not_drop_body(self):
        self.md.write_text('Contents\n参考文献\nAbstract\nSCIENCE\n## References\nCitations', encoding='utf-8')
        kept = common.turnIntoPureText(self.md)
        self.assertIn('SCIENCE', kept)
        self.assertNotIn('Citations', kept)

    def test_null_response_records_failure_and_fails_stage(self):
        with patch.object(examiner, 'single_chat', return_value=(None, 1)):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                examiner.run_examiner_for_domain('aqua', self.cfg, self.models, {}, self.run)
        invalid = list(Path(self.cfg['examiner_output']).rglob('*_invalid.txt'))
        self.assertEqual(len(invalid), 6)
        self.assertTrue(all(p.read_text(encoding='utf-8') for p in invalid))

    def test_missing_value_is_not_cached_as_success(self):
        with patch.object(examiner, 'single_chat', return_value=('{}', 1)):
            with self.assertRaises(RuntimeError):
                examiner.run_examiner_for_domain('aqua', self.cfg, self.models, {}, self.run)
        self.assertEqual(list(Path(self.cfg['examiner_output']).rglob('*.csv')), [])

    def test_request_failure_is_visible_to_cli(self):
        config = SimpleNamespace(DOMAINS={'aqua': self.cfg}, FULL_LLM_POOL=self.models,
                                 SELECTED_KEYS=list(self.models), LLM_SETTINGS={'mock': {}},
                                 RUN=self.run, EMBEDDING_MODEL={})
        with patch.object(examiner, 'single_chat', side_effect=RuntimeError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                run_pipeline.run('aqua', 'examiner', config)

    def test_duplicate_numeric_paper_ids_fail_before_api(self):
        (Path(self.cfg['markdown_dir']) / '01_other.md').write_text('Other body', encoding='utf-8')
        with patch.object(examiner, 'single_chat') as api:
            with self.assertRaisesRegex(ValueError, 'duplicate|collision'):
                examiner.run_examiner_for_domain('aqua', self.cfg, self.models, {}, self.run)
        api.assert_not_called()

    def test_changed_paper_reextracts_but_unchanged_resume_has_no_requests(self):
        self.extract()
        with patch.object(examiner, 'single_chat', side_effect=self.response) as api, patch.object(examiner, 'sleep_for_rate_limit'):
            examiner.run_examiner_for_domain('aqua', self.cfg, self.models, {'mock': {}}, self.run)
            self.assertEqual(api.call_count, 0)
            self.md.write_text('Changed paper content. China.', encoding='utf-8')
            examiner.run_examiner_for_domain('aqua', self.cfg, self.models, {'mock': {}}, self.run)
            self.assertEqual(api.call_count, 6)

    def test_rounds_are_isolated_in_composite(self):
        self.extract()
        self.run['round_index'] = 2
        self.extract()
        self.compose()
        df = pd.read_excel(Path(self.cfg['composite_dir']) / 'C_Q01.xlsx', dtype={'paper_index': str})
        self.assertEqual(len(df), 2)
        self.assertEqual(set(df.round_index), {2})

    def test_cross_resume_has_no_embedding_or_chat_calls(self):
        self.extract(); self.compose(); self.cross()
        embeds, queries = self.cross()
        self.assertEqual((embeds, queries), (0, 0))

    def test_cross_aggregation_is_idempotent(self):
        self.extract(); self.compose(); self.cross(); self.aggregate()
        path = Path(self.cfg['composite_dir']) / 'C_Q01.xlsx'
        once = pd.read_excel(path, dtype={'paper_index': str})
        self.aggregate()
        twice = pd.read_excel(path, dtype={'paper_index': str})
        pd.testing.assert_frame_equal(once, twice)
        self.assertEqual(list(twice.columns).count('cross_score'), 1)
        self.assertEqual(set(twice.cross_score), {1})

    def test_non_numeric_paper_ids_survive_all_stages(self):
        self.md.rename(self.md.with_name('Smith_2020.md'))
        self.extract(); self.compose(); self.cross(); self.aggregate()
        ensemble.run_ensemble_for_domain('aqua', self.cfg, self.run)
        df = pd.read_excel(Path(self.cfg['ensemble_dir']) / '00_full/aqua_Ensemble_Result_Q01_full.xlsx')
        self.assertEqual(set(df.paper_index), {'Smith_2020'})

    def test_changed_evidence_does_not_inherit_old_votes(self):
        self.extract(); self.compose(); self.cross(); self.aggregate()
        path = Path(self.cfg['composite_dir']) / 'C_Q01.xlsx'
        df = pd.read_excel(path, dtype={'paper_index': str})
        df.loc[0, 'evidence'] = 'Changed evidence'
        df.to_excel(path, index=False)
        self.aggregate()
        current = pd.read_excel(path)
        self.assertTrue(pd.isna(current.loc[0, 'cross_score']))
        self.assertEqual(current.loc[1, 'cross_score'], 1)

    def test_zero_threshold_rows_replace_old_result(self):
        self.extract(); self.compose(); self.cross(); self.aggregate()
        ensemble.run_ensemble_for_domain('aqua', self.cfg, self.run)
        path = Path(self.cfg['composite_dir']) / 'C_Q01.xlsx'
        df = pd.read_excel(path); df['cross_score'] = 0
        for col in [c for c in df if c.startswith('flag_')]:
            df.loc[df[col].notna(), col] = 0
        df.to_excel(path, index=False)
        ensemble.run_ensemble_for_domain('aqua', self.cfg, self.run)
        result = Path(self.cfg['ensemble_dir']) / 'MiniCross01/aqua_Ensemble_Result_Q01_MiniCross01.xlsx'
        self.assertTrue(pd.read_excel(result).empty)

    def test_empty_composite_supersedes_old_data(self):
        self.extract(); self.compose()
        self.models = {}; self.names = []
        self.compose()
        self.assertTrue(pd.read_excel(Path(self.cfg['composite_dir']) / 'C_Q01.xlsx').empty)

    def test_failed_composite_does_not_replace_last_complete_batch(self):
        self.extract(); self.compose()
        files = sorted(Path(self.cfg['composite_dir']).glob('*.xlsx'))
        previous = {file: file.read_bytes() for file in files}
        output = next(iter(examiner.expected_tasks('aqua', self.cfg, self.models, {'mock': {}}, self.run)))
        Path(output).write_text('truncated', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'composite failed'):
            self.compose()
        self.assertEqual({file: file.read_bytes() for file in files}, previous)

    def test_missing_current_cross_evidence_fails_instead_of_empty_success(self):
        self.extract(); self.compose(); self.aggregate()
        with self.assertRaisesRegex(ValueError, 'cross'):
            ensemble.run_ensemble_for_domain('aqua', self.cfg, self.run)


if __name__ == '__main__':
    unittest.main()
