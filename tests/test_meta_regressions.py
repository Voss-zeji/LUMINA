"""Regressions for lumina.ensemble_utils_meta (metadata ensemble).

Uses the real ensemble APIs (no stubs) and real pandas.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumina import ensemble_utils_meta as eum  # noqa: E402


def _rows(item: str, values, question_index: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"paper_index": 1, "question_index": question_index, "item": item, "value": v}
            for v in values
        ]
    )


def _ensemble(item: str, values, **kwargs) -> pd.Series:
    out = eum.ensemble_dataframe(_rows(item, values), **kwargs)
    assert len(out) == 1, f"expected one group row, got {len(out)}"
    return out.iloc[0]


class MissingValueTests(unittest.TestCase):
    """pandas NaN/NA/NaT and textual sentinels count as empty, not as text."""

    def test_float_nan_is_missing_not_the_word_nan(self) -> None:
        self.assertEqual(eum._normalize_single_value(float("nan"), "Study Location"), "")

    def test_pd_na_and_nat_are_missing(self) -> None:
        self.assertEqual(eum._normalize_single_value(pd.NA, "Study Location"), "")
        self.assertEqual(eum._normalize_single_value(pd.NaT, "Study Location"), "")

    def test_textual_sentinels_are_missing(self) -> None:
        for sentinel in ("N/A", "n/a", "NA", "nan", "NaT", "none", "null", "-", "--", "Not provided", "Not specified"):
            with self.subTest(sentinel=sentinel):
                self.assertEqual(eum._normalize_single_value(sentinel, "Study Location"), "")

    def test_nan_does_not_win_a_text_vote_as_a_token(self) -> None:
        detail = eum.ensemble_dataframe_all_data(_rows("Study Location", [float("nan"), "Beijing"]))
        self.assertNotIn("nan", set(detail["normalized_value"]))

    def test_nan_majority_yields_empty_ensemble_value(self) -> None:
        row = _ensemble("Study Location", [float("nan"), pd.NA, "Beijing"])
        self.assertTrue(math.isnan(row["ensemble_value"]))
        self.assertEqual(row["method"], "majority_empty")
        self.assertEqual(row["support"], 2)
        self.assertEqual(row["n_models"], 3)

    def test_real_answers_still_win_a_text_vote(self) -> None:
        row = _ensemble("Study Location", ["Beijing", float("nan"), "Beijing"])
        self.assertEqual(row["ensemble_value"], "beijing")
        self.assertEqual(row["support"], 2)

    def test_coord_missing_majority_is_empty(self) -> None:
        row = _ensemble("Longitude", [float("nan"), None, "119.5"])
        self.assertTrue(math.isnan(row["ensemble_value"]))
        # 与既有行为一致：coord 分支仍会追加 coord_round 后缀，不改方法学
        self.assertEqual(row["method"], "majority_empty+coord_round2")
        self.assertEqual(row["support"], 2)
        self.assertEqual(row["n_models"], 3)

    def test_pd_na_coord_endpoint_does_not_raise(self) -> None:
        row = _ensemble("Longitude", [pd.NA, pd.NA, pd.NA])
        self.assertTrue(math.isnan(row["ensemble_value"]))

    def test_period_sentinel_does_not_crash(self) -> None:
        row = _ensemble("Study Period", ["-", float("nan"), "2011"])
        self.assertTrue(math.isnan(row["ensemble_value"]))
        self.assertEqual(row["method"], "majority_empty")

    def test_all_sentinel_period_group_does_not_crash(self) -> None:
        row = _ensemble("Study Period", ["-", "NA"])
        self.assertTrue(math.isnan(row["ensemble_value"]))


class SignedCoordinateRangeTests(unittest.TestCase):
    """A leading '-' is a sign, not a range separator."""

    def test_negative_word_range_keeps_its_sign(self) -> None:
        self.assertEqual(eum._parse_coord_interval("-74.0 to -73.0"), (-74.0, -73.0))

    def test_negative_to_positive_dash_range_preserves_baseline_meaning(self):
        self.assertEqual(eum._parse_coord_interval('-74-73'), (-74., 73.))
        self.assertEqual(eum._parse_coord_interval('-74–73'), (-74., 73.))
        self.assertEqual(eum._parse_coord_interval('-74 –73'), (-74., 73.))

    def test_range_hemisphere_suffix_applies_to_both_endpoints(self):
        self.assertEqual(eum._parse_coord_interval('33.5-32.0S'), (-33.5, -32.))
        self.assertEqual(eum._parse_coord_interval('33.5S-32.0S'), (-33.5, -32.))

    def test_negative_bare_dash_range_keeps_its_sign(self) -> None:
        self.assertEqual(eum._parse_coord_interval("-74 -73"), (-74.0, -73.0))

    def test_hemisphere_suffix_agrees_with_signed_range(self) -> None:
        self.assertEqual(eum._parse_coord_interval("74W to 73W"), (-74.0, -73.0))

    def test_hemisphere_suffix_on_both_ends_of_a_dash_range(self) -> None:
        self.assertEqual(eum._parse_coord_interval("73.5S-74.5S"), (-74.5, -73.5))
        self.assertEqual(eum._parse_coord_interval("120.0E-119.5E"), (119.5, 120.0))

    def test_positive_word_range_is_unchanged(self) -> None:
        self.assertEqual(eum._parse_coord_interval("119.5 to 120.0"), (119.5, 120.0))
        self.assertEqual(eum._parse_coord_interval("119.5-120.0"), (119.5, 120.0))

    def test_dms_range_is_unchanged(self) -> None:
        self.assertEqual(
            eum._parse_coord_interval("119°30'00\"E to 120°00'00\"E"), (119.5, 120.0)
        )

    def test_negative_range_output_is_negative_end_to_end(self) -> None:
        row = _ensemble("Longitude", ["-74.0 to -73.0"] * 3)
        self.assertEqual(row["ensemble_value"], "-74.00–-73.00")
        self.assertEqual(row["support"], 3)

    def test_signed_and_hemisphere_answers_vote_together(self) -> None:
        row = _ensemble("Latitude", ["-34.9 to -34.8", "34.9S to 34.8S"])
        self.assertEqual(row["ensemble_value"], "-34.90–-34.80")
        self.assertEqual(row["support"], 2)

    def test_single_negative_point_is_unchanged(self) -> None:
        self.assertEqual(eum._parse_coord_interval("-34.9"), (-34.9, -34.9))


class IsoStudyPeriodTests(unittest.TestCase):
    """ISO dates parse before the dash range split eats their hyphens."""

    def test_iso_day_range(self) -> None:
        p = eum._merge_blocks_to_period("2011-01-01 to 2011-06-30")
        self.assertEqual(eum._format_period(p), "2011-01-01 to 2011-06-30")

    def test_iso_day_range_with_dash(self) -> None:
        p = eum._merge_blocks_to_period("2011-01-01 - 2011-06-30")
        self.assertEqual(eum._format_period(p), "2011-01-01 to 2011-06-30")

    def test_iso_day_range_with_leading_from(self) -> None:
        p = eum._merge_blocks_to_period("from 2011-01-01 to 2011-06-30")
        self.assertEqual(eum._format_period(p), "2011-01-01 to 2011-06-30")

    def test_iso_month_range(self) -> None:
        p = eum._merge_blocks_to_period("2011-01 to 2011-06")
        self.assertEqual(eum._format_period(p), "2011-01 to 2011-06")

    def test_single_iso_day(self) -> None:
        p = eum._merge_blocks_to_period("2011-03-14")
        self.assertEqual(eum._format_period(p), "2011-03-14")

    def test_iso_and_month_name_ranges_vote_together(self) -> None:
        row = _ensemble("Study Period", ["2011-01-01 to 2011-06-30", "January 1, 2011 to June 30, 2011"])
        self.assertEqual(row["ensemble_value"], "2011-01-01 to 2011-06-30")
        self.assertEqual(row["support"], 2)

    def test_year_range_is_not_mistaken_for_iso_month(self) -> None:
        p = eum._merge_blocks_to_period("2011-2012")
        self.assertEqual(eum._format_period(p), "2011 to 2012")


class PreservedVotePolicyTests(unittest.TestCase):
    """The fixes must not move majority/tie/precision/period policies."""

    def test_precision_two_merges_close_coordinates(self) -> None:
        row = _ensemble("Longitude", ["119.6279", "119.628", "119.63"], precision_digits=2)
        self.assertEqual(row["ensemble_value"], "119.63")
        self.assertEqual(row["support"], 3)

    def test_precision_four_keeps_them_apart(self) -> None:
        row = _ensemble("Longitude", ["119.6279", "119.628", "119.63"], precision_digits=4)
        # 既有行为：并列时按 (跨度, 距中位数中心距离, 字符串) 排序后取第一个，
        # 该顺序本身不在本次修复范围内，故保持基线取值。
        self.assertEqual(row["ensemble_value"], "119.6280")
        self.assertEqual(row["support"], 1)

    def test_exactly_half_empty_is_not_a_majority(self) -> None:
        row = _ensemble("Study Location", ["Beijing", "Beijing", "Shanghai", None])
        self.assertEqual(row["ensemble_value"], "beijing")
        self.assertEqual(row["support"], 2)
        self.assertEqual(row["n_models"], 4)

    def test_year_precision_majority_wins_a_tie(self) -> None:
        row = _ensemble("Study Period", ["2011", "2011-01-01", "2011"])
        self.assertEqual(row["ensemble_value"], "2011")
        self.assertEqual(row["support"], 2)

    def test_month_range_same_year_is_unchanged(self) -> None:
        for text in ("May-December 2010", "May to December 2010"):
            with self.subTest(text=text):
                self.assertEqual(
                    _ensemble("Study Period", [text] * 3)["ensemble_value"], "2010-05 to 2010-12"
                )

    def test_long_date_range_is_unchanged(self) -> None:
        row = _ensemble("Study Period", ["December 10, 2011 to January 20, 2012"] * 2)
        self.assertEqual(row["ensemble_value"], "2011-12-10 to 2012-01-20")

    def test_all_data_fraction_unchanged(self) -> None:
        detail = eum.ensemble_dataframe_all_data(
            _rows("Study Location", ["Beijing (China)", "China, Beijing"])
        )
        self.assertEqual(sorted(detail["normalized_value"]), ["beijing", "beijing china"])
        self.assertEqual(sorted(detail["fraction"]), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
