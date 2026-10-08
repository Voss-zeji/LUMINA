from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from lumina import ensemble, evaluation, prompts
from lumina.common import candidate_id, write_dataframe


class DualThresholdConsensus(unittest.TestCase):
    def publish(self, rows, *, domain="aqua", question=3, models=("A", "B", "C"),
                verification=2, consensus=2):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.latest_root = root
        cfg = dict(composite_dir=str(root / "composite"), composite_prefix="LUMINA",
                   ensemble_dir=str(root / "ensemble"))
        records = []
        for index, values in enumerate(rows):
            row = dict(paper_index="01", question_index=question, round_index=1,
                       item="Flux-1" if question != 1 else "Study_location", value=2.12,
                       unit="mg m-2 h-1", evidence=f"Table {index + 1}", confidence_lv=95,
                       model="A", paper_fingerprint="paper", request_fingerprint="request",
                       cross_score=verification)
            row.update(values)
            other = [model for model in models if model != row["model"]]
            for model in models:
                row[f"flag_{model}"] = (float("nan") if model == row["model"] else
                                        int(other.index(model) < row["cross_score"]))
            row["candidate_id"] = candidate_id(row)
            records.append(row)
        frame = pd.DataFrame(records)
        for q in range(1, len(prompts.questions_for_domain(domain)) + 1):
            write_dataframe(frame if q == question else frame.iloc[:0],
                            root / "composite" / f"LUMINA_Q{q:02d}.xlsx")
        run = dict(min_cross_scores=[verification], min_consensus_models=consensus)
        if consensus is None:
            run.pop('min_consensus_models')
        ensemble.run_ensemble_for_domain(domain, cfg, run, list(models))
        def read(folder, kind, label):
            return pd.read_excel(root / "ensemble" / folder /
                                 f"{domain}_Ensemble_{kind}_Q{question:02d}_{label}.xlsx",
                                 keep_default_na=False)
        label = f"MiniCross{verification:02d}"
        return (read(label, "Result", label), read(label, "All-Standard-Answers", label),
                read("00_full", "Result", "full"))

    def test_verification_pass_alone_does_not_accept_a_single_generator(self):
        result, _, _ = self.publish([dict(model="A")])
        self.assertTrue(result.empty, "T_vfy passing cannot substitute for T_bsl")

    def test_unverified_nan_score_cannot_pass_standalone_confirmation(self):
        frame = pd.DataFrame([dict(paper_index='01', item='Flux-1', value=2.12,
                                   evidence='Table 1', model='A', cross_score=float('nan'))])
        result, _ = ensemble.confirm_consensus('aqua', 3, frame, ['A', 'B', 'C'], 2, 1)
        self.assertTrue(result.empty)

    def test_consensus_pass_alone_does_not_accept_failed_verification(self):
        result, _, _ = self.publish([dict(model=m, cross_score=1) for m in "ABC"])
        self.assertTrue(result.empty)

    def test_both_gates_accept_matching_distinct_models(self):
        result, _, _ = self.publish([dict(model="A"), dict(model="B"),
                                     dict(model="C", value=3.34, cross_score=1)])
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]["vote_count"]), 2)
        self.assertTrue(result.iloc[0]["accepted"])
        self.assertEqual(result.iloc[0]["aggregation_mode"], "dual_threshold")
        self.assertEqual(int(result.iloc[0]["consensus_threshold"]), 2)

    def test_repeated_rows_from_one_model_cannot_create_consensus(self):
        result, _, _ = self.publish([dict(model="A") for _ in range(8)])
        self.assertTrue(result.empty, "eight rows from one model are still one model")

    def test_decimal_values_do_not_bypass_mode_and_consensus(self):
        result, _, _ = self.publish([dict(model="A"), dict(model="B"),
                                     dict(model="C", value=3.34)])
        self.assertEqual(len(result), 1)
        self.assertEqual(float(result.iloc[0]["ensemble_value"]), 2.12)

    def test_even_with_threshold_one_only_the_supported_mode_is_accepted(self):
        result, _, _ = self.publish([dict(model="A"), dict(model="B"),
                                     dict(model="C", value=3.34)], consensus=1)
        self.assertEqual(len(result), 1)

    def test_tied_modes_remain_unaccepted_instead_of_confidence_tiebreak(self):
        result, detail, _ = self.publish(
            [dict(model="A", value=2), dict(model="B", value=2),
             dict(model="C", value=3), dict(model="D", value=3, confidence_lv=100)],
            models=("A", "B", "C", "D"), verification=3)
        self.assertTrue(result.empty)
        self.assertEqual(set(detail["gate_reason"]), {"tied_mode"})

    def test_different_units_cannot_share_baseline_votes(self):
        result, _, _ = self.publish([dict(model="A"),
                                     dict(model="B", unit="g m-2 h-1")])
        self.assertTrue(result.empty)

    def test_different_targets_cannot_share_votes_for_the_same_decimal(self):
        result, _, _ = self.publish([dict(model="A", item="Flux-1"),
                                     dict(model="B", item="Flux-2")])
        self.assertTrue(result.empty)

    def test_distinct_experiment_markers_are_not_combined(self):
        result, _, _ = self.publish([dict(model="A", experimental="expA"),
                                     dict(model="B", experimental="expB")])
        self.assertTrue(result.empty)

    def test_metadata_also_requires_distinct_model_consensus(self):
        result, _, _ = self.publish([dict(model="A", value="China")], question=1)
        self.assertTrue(result.empty)

    def test_coordinate_precision_normalization_preserves_agreement(self):
        result, _, _ = self.publish([dict(model="A", item="Latitude", value=45.00001),
                                     dict(model="B", item="Latitude", value=45.00002)],
                                    question=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(float(result.iloc[0]["ensemble_value"]), 45)
        self.assertEqual(int(result.iloc[0]["support"]), 2)

    def test_full_results_are_explicitly_diagnostic_not_accepted(self):
        _, _, full = self.publish([dict(model="A")])
        self.assertFalse(full["accepted"].any())
        self.assertEqual(set(full["aggregation_mode"]), {"diagnostic"})

    def test_unreachable_or_noninteger_consensus_is_rejected(self):
        for threshold in (0, 4, True, 1.5):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(
                    ValueError, "min_consensus_models"):
                self.publish([dict(model="A")], consensus=threshold)

    def test_default_majority_uses_full_pool_not_surviving_models(self):
        result, detail, _ = self.publish([dict(model='A'), dict(model='B')],
                                         models=('A', 'B', 'C', 'D', 'E'), consensus=None)
        self.assertTrue(result.empty)
        self.assertEqual(set(detail.consensus_threshold), {3})
        self.assertEqual(set(detail.gate_reason), {'below_consensus_threshold'})

    def test_only_verification_passing_model_votes_count_for_consensus(self):
        result, _, _ = self.publish([dict(model='A'), dict(model='B', cross_score=1)])
        self.assertTrue(result.empty)

    def test_wildfire_numeric_and_aqua_species_use_the_same_two_gates(self):
        for domain, question, item, value in [('wildfire', 2, 'Mass', 2.12),
                                               ('aqua', 2, 'Specie', 'Tilapia')]:
            with self.subTest(domain=domain):
                result, _, _ = self.publish([dict(model='A', item=item, value=value)],
                                             domain=domain, question=question)
                self.assertTrue(result.empty)

    def test_evaluation_keeps_winner_identity_and_inline_unit(self):
        self.publish([dict(model='A', experimental='expA', value='2.120 mg', unit=''),
                      dict(model='B', experimental='expA', value='2.120 mg', unit=''),
                      dict(model='C', experimental='expB', value='2.120 mg', unit='')])
        frame = pd.read_excel(self.latest_root / 'composite/LUMINA_Q03.xlsx', keep_default_na=False)
        raw = [dict(row, _round=1, _question=3) for row in frame.to_dict('records')]
        path = self.latest_root / 'ensemble/MiniCross02/aqua_Ensemble_Result_Q03_MiniCross02.xlsx'
        records = evaluation._result_rows(path, 1, 'MiniCross02', 'aqua', raw)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['method'], 'winner')
        self.assertEqual(records[0]['source_experiments'], ['expA'])
        self.assertEqual(records[0]['experiment_id'], 'expA')
        self.assertEqual(records[0]['unit'], 'mg')
        self.assertIsNone(records[0]['candidate_set'])


if __name__ == "__main__":
    unittest.main()
