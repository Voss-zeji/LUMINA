# -*- coding: utf-8 -*-
"""数值集成的回归测试（真实 pandas，无 stub / 无网络）。

覆盖审计 F02（范围/科学计数法/单位指数解析）与 F03（独立 unit 字段被消费并保留），
并锁定必须保持不变的既有策略：逐 item 投票、tie-break、小数多数模式的跨 item 候选策略、
无意义值过滤、非数值 confidence 的等权处理。
"""

from __future__ import annotations

import unittest
import warnings

import pandas as pd

from lumina.ensemble_utils_value import (
    ensemble_numeric_dataframe,
    ensemble_numeric_dataframe_all_data,
    normalize_numeric_string,
    normalize_numeric_with_unit,
    summarize_all_value_options,
    summarize_value_majorities_by_numeric,
    summarize_value_options,
)


def _df(rows):
    return pd.DataFrame(rows)


class TestNumericParsing(unittest.TestCase):
    """F02：范围连字符、科学计数法与单位指数不得互相污染。"""

    def test_unsigned_hyphen_range_keeps_sign(self):
        # 旧实现把 "10-20" 解析成 "-20-10"
        self.assertEqual(normalize_numeric_string("10-20"), "10-20")
        self.assertEqual(normalize_numeric_string("10–20"), "10-20")
        self.assertEqual(normalize_numeric_string("10 ~ 20"), "10-20")
        self.assertEqual(normalize_numeric_string("10 to 20"), "10-20")

    def test_hyphen_range_and_word_range_share_one_bucket(self):
        self.assertEqual(normalize_numeric_string("10-20"), normalize_numeric_string("10 to 20"))
        self.assertEqual(normalize_numeric_string("20-10"), "10-20")
        # 范围词不得残留进单位（否则同一范围会分裂成多个票箱）
        self.assertEqual(normalize_numeric_with_unit("10-20"), "10-20")
        self.assertEqual(normalize_numeric_with_unit("10 to 20"), "10-20")
        self.assertEqual(normalize_numeric_with_unit("10~20"), "10-20")
        self.assertEqual(normalize_numeric_with_unit("1e-3"), "0.001")

    def test_signed_range_keeps_endpoint_signs(self):
        self.assertEqual(normalize_numeric_string("-10 to -20"), "-20--10")
        self.assertEqual(normalize_numeric_string("-10 to 20"), "-10-20")
        self.assertEqual(normalize_numeric_string("+5 to +10"), "5-10")

    def test_scientific_notation_is_not_a_range(self):
        self.assertEqual(normalize_numeric_string("1e-3"), "0.001")
        self.assertEqual(normalize_numeric_string("2.5e-4"), "0.00025")
        self.assertEqual(normalize_numeric_string("1E3"), "1000")

    def test_unit_exponent_does_not_become_a_range_endpoint(self):
        # 本 pipeline 的标准通量单位：旧实现得到 "-2-0.01 kgc m-2 yr-1"
        self.assertEqual(normalize_numeric_string("0.01 kgC m-2 yr-1"), "0.01")
        self.assertEqual(normalize_numeric_string("1.5 ha yr-1"), "1.5")
        self.assertEqual(normalize_numeric_string("2.5e-4 m"), "0.00025")

    def test_pm_keeps_center_value(self):
        self.assertEqual(normalize_numeric_string("12.3 ± 0.4"), "12.3")

    def test_unparseable_still_returns_none(self):
        self.assertIsNone(normalize_numeric_string("not provided"))
        self.assertIsNone(normalize_numeric_string("N/A"))

    def test_inline_unit_handling_is_preserved(self):
        # 旧的内联单位行为：数值 + 小写去重排序后的单位 token
        self.assertEqual(normalize_numeric_with_unit("0.01 kgC m-2 yr-1"), "0.01 kgc m-2 yr-1")
        # 规范单位为 token 排序后的形式，且缺失单位不改变数值部分
        self.assertEqual(normalize_numeric_with_unit("2 m-2"), "2 m-2")
        self.assertEqual(normalize_numeric_with_unit("2 m-2 kg"), "2 kg m-2")
        self.assertEqual(normalize_numeric_with_unit("2 kg m-2"), normalize_numeric_with_unit("2 m-2 kg"))
        self.assertEqual(normalize_numeric_with_unit("2"), "2")


class TestUnitColumnContract(unittest.TestCase):
    """F03：producer 的独立 unit 字段必须被消费并保留，不同单位不得合并。"""

    def test_different_explicit_units_never_merge(self):
        df = _df([
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "mg m-2 h-1", "model": "m1"},
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "g m-2 h-1", "model": "m2"},
        ])
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertEqual(len(detail), 2, "不同显式单位必须保留为两个候选，不得合并计票")
        self.assertEqual(set(detail["unit"]), {"h-1 m-2 mg", "g h-1 m-2"})
        self.assertEqual(set(detail["count"]), {1})

    def test_unit_token_order_does_not_split_votes(self):
        df = _df([
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "mg m-2 h-1", "model": "m1"},
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "m-2 h-1 mg", "model": "m2"},
        ])
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertEqual(len(detail), 1)
        self.assertEqual(int(detail.iloc[0]["count"]), 2)
        self.assertEqual(detail.iloc[0]["unit"], "h-1 m-2 mg")

    def test_same_explicit_unit_merges_and_unit_is_preserved(self):
        df = _df([
            {"paper_index": 1, "item": "Flux-1", "value": "2.00", "unit": "mg m-2 h-1", "model": "m1"},
            {"paper_index": 1, "item": "Flux-1", "value": "2", "unit": "mg m-2 h-1", "model": "m2"},
        ])
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertEqual(len(detail), 1)
        self.assertEqual(int(detail.iloc[0]["count"]), 2)
        self.assertEqual(detail.iloc[0]["unit"], "h-1 m-2 mg")

    def test_no_automatic_physical_conversion(self):
        # 1 kg 与 1000 g 物理等价，但禁止自动换算 -> 必须是两个候选
        df = _df([
            {"paper_index": 1, "item": "Biomass", "value": 1, "unit": "kg", "model": "m1"},
            {"paper_index": 1, "item": "Biomass", "value": 1000, "unit": "g", "model": "m2"},
        ])
        self.assertEqual(len(ensemble_numeric_dataframe_all_data(df)), 2)

    def test_si_prefix_case_is_significant(self):
        # mW(毫瓦)/MW(兆瓦)/uW(微瓦) 是不同的物理单位，绝不能合并
        df = _df([
            {"paper_index": 1, "item": "Power", "value": 2, "unit": "mW", "model": "m1"},
            {"paper_index": 1, "item": "Power", "value": 2, "unit": "MW", "model": "m2"},
            {"paper_index": 1, "item": "Power", "value": 2, "unit": "µW", "model": "m3"},
        ])
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertEqual(len(detail), 3)
        self.assertEqual(set(detail["count"]), {1})
        self.assertEqual(set(detail["unit"]), {"mW", "MW", "µW"})

    def test_unicode_unit_syntax_is_preserved(self):
        # prompts.py 的 Aqua Q3 例子里就是这种排版单位。
        # 不变量：排版分隔符（间隔号/点）被规整，但 ¯/²/¹ 等指数字符不得丢失。
        expected = {
            "mg・m¯².h¯¹": "h¯¹ mg m¯²",
            "mg m¯² h¯¹": "h¯¹ mg m¯²",
            "μg m-3": "m-3 μg",
        }
        for unit, want in expected.items():
            detail = ensemble_numeric_dataframe_all_data(_df([
                {"paper_index": 1, "item": "F", "value": 2, "unit": unit, "model": "m1"},
            ]))
            self.assertEqual(detail.iloc[0]["unit"], want)
            self.assertEqual(detail.iloc[0]["normalized_value"], f"2 {want}")

    def test_unicode_units_of_same_measure_merge(self):
        # 同一物理量的两种排版（间隔号 vs 空格）应合并，不分裂票箱
        a = ensemble_numeric_dataframe_all_data(_df([
            {"paper_index": 1, "item": "F", "value": 2, "unit": "mg・m¯².h¯¹", "model": "m1"},
        ]))
        b = ensemble_numeric_dataframe_all_data(_df([
            {"paper_index": 1, "item": "F", "value": 2, "unit": "mg m¯² h¯¹", "model": "m2"},
        ]))
        self.assertEqual(a.iloc[0]["unit"], b.iloc[0]["unit"])

    def test_final_winner_is_scalar_plus_unit(self):
        # 同一 item、同一显式单位：赢家为标量数值 + 独立 unit 列
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "unit": "mg m-2 h-1", "model": "m1", "confidence_lv": 90},
            {"paper_index": 1, "item": "F", "value": 3, "unit": "mg m-2 h-1", "model": "m2", "confidence_lv": 90},
            {"paper_index": 1, "item": "F", "value": 3, "unit": "mg m-2 h-1", "model": "m3", "confidence_lv": 90},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["ensemble_value"], "3")
        self.assertEqual(res.iloc[0]["unit"], "h-1 m-2 mg")
        self.assertEqual(int(res.iloc[0]["vote_count"]), 2)

    def test_unit_column_wins_and_missing_falls_back_to_inline(self):
        explicit = ensemble_numeric_dataframe_all_data(_df([
            {"paper_index": 1, "item": "F", "value": "0.01 kgC m-2 yr-1", "unit": "g C m-2 yr-1", "model": "m1"},
        ]))
        self.assertEqual(explicit.iloc[0]["unit"], "C g m-2 yr-1")
        self.assertEqual(explicit.iloc[0]["normalized_value"], "0.01 C g m-2 yr-1")

        inline = ensemble_numeric_dataframe_all_data(_df([
            {"paper_index": 1, "item": "F", "value": "0.01 kgC m-2 yr-1", "model": "m1"},
        ]))
        self.assertEqual(inline.iloc[0]["unit"], "kgc m-2 yr-1")

        nan_unit = ensemble_numeric_dataframe_all_data(_df([
            {"paper_index": 1, "item": "F", "value": "0.01 kgC m-2 yr-1", "unit": None, "model": "m1"},
        ]))
        self.assertEqual(nan_unit.iloc[0]["unit"], "kgc m-2 yr-1")

    def test_result_frame_keeps_unit(self):
        df = _df([
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "mg m-2 h-1", "model": "m1"},
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "g m-2 h-1", "model": "m2"},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertIn("unit", res.columns)
        # 同一 item 仍只输出一个赢家，但其单位必须随结果保留
        self.assertEqual(len(res), 1)
        self.assertIn(res.iloc[0]["unit"], {"h-1 m-2 mg", "g h-1 m-2"})

    def test_explicit_unit_result_is_scalar_plus_unit_column(self):
        # 显式 unit 列 -> ensemble_value 必须是标量数值，单位只出现在 unit 列
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "unit": "mg m-2 h-1", "model": "m1", "confidence_lv": 90},
            {"paper_index": 1, "item": "F", "value": 2, "unit": "mg m-2 h-1", "model": "m2", "confidence_lv": 90},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(res.iloc[0]["ensemble_value"], "2")
        self.assertEqual(res.iloc[0]["unit"], "h-1 m-2 mg")
        self.assertTrue(bool(res.iloc[0]["unit_is_explicit"]))

    def test_inline_unit_result_keeps_legacy_joint_string(self):
        # 无 unit 列 -> 旧的联合字符串行为必须保持不变
        df = _df([
            {"paper_index": 1, "item": "F", "value": "0.01 kgC m-2 yr-1", "model": "m1", "confidence_lv": 90},
            {"paper_index": 1, "item": "F", "value": "0.01 kgC m-2 yr-1", "model": "m2", "confidence_lv": 90},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(res.iloc[0]["ensemble_value"], "0.01 kgc m-2 yr-1")
        self.assertFalse(bool(res.iloc[0]["unit_is_explicit"]))

    def test_decimal_majority_mode_uses_scalar_for_explicit_unit(self):
        # 小数多数模式的候选策略不变，仅显式 unit 行的取值改为标量
        df = _df([
            {"paper_index": 1, "item": "a_biomass", "value": "0.51", "unit": "kg", "model": "m1"},
            {"paper_index": 1, "item": "b_biomass", "value": "0.51", "unit": "kg", "model": "m2"},
            {"paper_index": 1, "item": "c_biomass", "value": "0.22", "unit": "kg", "model": "m3"},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(sorted(res["ensemble_value"]), ["0.22", "0.51"])  # 策略不变
        winner = res[res["ensemble_value"] == "0.51"].iloc[0]
        self.assertEqual(int(winner["vote_count"]), 2)
        self.assertEqual(winner["unit"], "kg")

    def test_unit_conflict_splits_votes_in_result_frame(self):
        # 单位冲突时票数不再合并，赢家按既有 tie-break（count, conf, 字符串）选出
        df = _df([
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "mg m-2 h-1", "model": "m1", "confidence_lv": 50},
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "g m-2 h-1", "model": "m2", "confidence_lv": 50},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(len(res), 1)
        self.assertEqual(int(res.iloc[0]["vote_count"]), 1)
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertEqual(sorted(detail["count"]), [1, 1])

    def test_unit_survives_pipeline_normalization(self):
        df = _df([
            {"paper_index": 1, "question_index": 3, "item": "Flux-1", "value": "10-20", "unit": "mg m-2 h-1", "model": "m1"},
            {"paper_index": 1, "question_index": 3, "item": "Flux-1", "value": "10-20", "unit": "mg m-2 h-1", "model": "m2"},
        ])
        res = ensemble_numeric_dataframe(df)
        # 显式 unit -> 标量范围 + 独立 unit；明细仍保留联合键以供追溯
        self.assertEqual(res.iloc[0]["ensemble_value"], "10-20")
        self.assertEqual(int(res.iloc[0]["vote_count"]), 2)
        self.assertEqual(res.iloc[0]["unit"], "h-1 m-2 mg")
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertEqual(detail.iloc[0]["normalized_value"], "10-20 h-1 m-2 mg")
        self.assertEqual(detail.iloc[0]["scalar_value"], "10-20")

    def test_summarize_paths_also_honour_the_unit_column(self):
        df = _df([
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "mg m-2 h-1", "model": "m1"},
            {"paper_index": 1, "item": "Flux-1", "value": 2, "unit": "g m-2 h-1", "model": "m2"},
        ])
        self.assertEqual(len(summarize_value_options(df, 1, "Flux-1")), 2)
        self.assertEqual(len(summarize_all_value_options(df)), 2)
        self.assertEqual(len(summarize_value_majorities_by_numeric(df, 1)), 2)


class TestConfidencePolicy(unittest.TestCase):
    """非数值 confidence 必须被明确处理，且不得改变有效票的选择。"""

    def test_non_numeric_confidence_does_not_abort_and_is_equal_weight(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "model": "m1", "confidence_lv": "high"},
            {"paper_index": 1, "item": "F", "value": 2, "model": "m2", "confidence_lv": 90},
            {"paper_index": 1, "item": "F", "value": 3, "model": "m3", "confidence_lv": "N/A"},
        ])
        detail = ensemble_numeric_dataframe_all_data(df)
        got = {r["normalized_value"]: r for _, r in detail.iterrows()}
        self.assertEqual(int(got["2"]["count"]), 2)
        self.assertAlmostEqual(float(got["2"]["confidence_sum"]), 91.0)  # 1.0 兜底 + 90
        self.assertAlmostEqual(float(got["3"]["confidence_sum"]), 1.0)

        res = ensemble_numeric_dataframe(df)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["ensemble_value"], "2")
        self.assertEqual(int(res.iloc[0]["vote_count"]), 2)

    def test_invalid_confidence_is_reported_not_silently_dropped(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "model": "m1", "confidence_lv": "high"},
            {"paper_index": 1, "item": "F", "value": 2, "model": "m2", "confidence_lv": 90},
        ])
        with self.assertWarns(RuntimeWarning) as ctx:
            ensemble_numeric_dataframe_all_data(df)
        self.assertIn("1/2", str(ctx.warning))

    def test_valid_confidence_emits_no_warning(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "model": "m1", "confidence_lv": 90},
            {"paper_index": 1, "item": "F", "value": 2, "model": "m2", "confidence_lv": 80},
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ensemble_numeric_dataframe_all_data(df)

    def test_nan_confidence_is_equal_weight(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "model": "m1", "confidence_lv": None},
            {"paper_index": 1, "item": "F", "value": 2, "model": "m2", "confidence_lv": 10},
        ])
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertAlmostEqual(float(detail.iloc[0]["confidence_sum"]), 11.0)

    def test_valid_confidence_vote_selection_unchanged(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "model": "m1", "confidence_lv": 50},
            {"paper_index": 1, "item": "F", "value": 3, "model": "m2", "confidence_lv": 90},
            {"paper_index": 1, "item": "F", "value": 3, "model": "m3", "confidence_lv": 90},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(res.iloc[0]["ensemble_value"], "3")
        self.assertEqual(int(res.iloc[0]["vote_count"]), 2)

    def test_alias_confidence_column_still_recognized(self):
        df = _df([{"paper_index": 1, "item": "F", "value": 2, "model": "m1", "confidence": 70}])
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertAlmostEqual(float(detail.iloc[0]["confidence_sum"]), 70.0)


class TestRealEnsembleEntryPoint(unittest.TestCase):
    """通过真实的 ensemble._ensemble_subset 校验最终输出契约。"""

    def _rows(self, **kw):
        base = {"paper_index": 1, "question_index": 3, "item": "Flux-1", "confidence_lv": 90}
        base.update(kw)
        return base

    def test_explicit_unit_outputs_scalar_and_inline_keeps_legacy_string(self):
        from lumina import ensemble

        explicit = _df([
            self._rows(value=2, unit="mg m-2 h-1", model="m1"),
            self._rows(value=2, unit="mg m-2 h-1", model="m2"),
        ])
        res, detail = ensemble._ensemble_subset("aqua", 3, explicit)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["ensemble_value"], "2", "显式 unit 的最终值必须是标量")
        self.assertEqual(res.iloc[0]["unit"], "h-1 m-2 mg")
        # 明细保留联合键，来源可追溯
        self.assertEqual(detail.iloc[0]["normalized_value"], "2 h-1 m-2 mg")
        self.assertEqual(detail.iloc[0]["scalar_value"], "2")

        legacy = _df([
            self._rows(value="0.01 kgC m-2 yr-1", model="m1"),
            self._rows(value="0.01 kgC m-2 yr-1", model="m2"),
        ])
        res2, _ = ensemble._ensemble_subset("aqua", 3, legacy)
        self.assertEqual(res2.iloc[0]["ensemble_value"], "0.01 kgc m-2 yr-1")


class TestPreservedPolicies(unittest.TestCase):
    """既有策略必须保持不变。"""

    def test_meaningless_sentinels_are_dropped(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": "999999", "model": "m1"},
            {"paper_index": 1, "item": "F", "value": "-1", "model": "m2"},
            {"paper_index": 1, "item": "F", "value": "n/a", "model": "m3"},
        ])
        self.assertTrue(ensemble_numeric_dataframe_all_data(df).empty)
        self.assertTrue(ensemble_numeric_dataframe(df).empty)

    def test_per_item_vote_groups_by_item_equivalence(self):
        df = _df([
            {"paper_index": 1, "item": "aboveground_biomass", "value": 10, "model": "m1"},
            {"paper_index": 1, "item": "Aboveground biomass", "value": 10, "model": "m2"},
            {"paper_index": 1, "item": "belowground_biomass", "value": 20, "model": "m1"},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(len(res), 2)
        self.assertEqual(int(res[res["item"] == "belowground biomass"].iloc[0]["vote_count"]), 1)

    def test_tie_break_is_votes_then_confidence_then_string(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": 9, "model": "m1", "confidence_lv": 50},
            {"paper_index": 1, "item": "F", "value": 10, "model": "m2", "confidence_lv": 50},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["ensemble_value"], "9")  # 字符串 tie-break，字典序最大

    def test_confidence_preferred_when_requested(self):
        df = _df([
            {"paper_index": 1, "item": "F", "value": 2, "model": "m1", "confidence_lv": 10},
            {"paper_index": 1, "item": "F", "value": 3, "model": "m2", "confidence_lv": 10},
            {"paper_index": 1, "item": "F", "value": 3, "model": "m3", "confidence_lv": 90},
        ])
        res = ensemble_numeric_dataframe(df, prefer="confidence")
        self.assertEqual(res.iloc[0]["ensemble_value"], "3")

    def test_decimal_majority_mode_keeps_all_cross_item_candidates(self):
        # 既定策略：≥50% 有效数值有 ≥2 位小数时，跨 item 按数值保留全部候选
        df = _df([
            {"paper_index": 1, "item": "aboveground_biomass", "value": "0.51", "model": "m1"},
            {"paper_index": 1, "item": "belowground_biomass", "value": "0.51", "model": "m2"},
            {"paper_index": 1, "item": "litter_biomass", "value": "0.22", "model": "m3"},
        ])
        res = ensemble_numeric_dataframe(df)
        self.assertEqual(sorted(res["ensemble_value"]), ["0.22", "0.51"])
        winner = res[res["ensemble_value"] == "0.51"].iloc[0]
        self.assertEqual(int(winner["vote_count"]), 2)

    def test_decimal_majority_threshold_can_be_disabled(self):
        df = _df([
            {"paper_index": 1, "item": "a_biomass", "value": "0.51", "model": "m1"},
            {"paper_index": 1, "item": "b_biomass", "value": "0.51", "model": "m2"},
        ])
        res = ensemble_numeric_dataframe(df, majority_decimal_ratio=1.1)
        self.assertEqual(len(res), 2)
        self.assertEqual(sorted(res["ensemble_value"]), ["0.51", "0.51"])

    def test_ilegal_items_are_dropped(self):
        df = _df([
            {"paper_index": 1, "item": "ilegal", "value": 5, "model": "m1"},
            {"paper_index": 1, "item": "F", "value": 5, "model": "m2"},
        ])
        detail = ensemble_numeric_dataframe_all_data(df)
        self.assertEqual(set(detail["item"]), {"F"})
        self.assertEqual(int(detail.iloc[0]["count"]), 1)


if __name__ == "__main__":
    unittest.main()
