from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_feature_trial import (  # noqa: E402
    APP_OPENS_PER_SCREEN_HOUR,
    ENTERTAINMENT_SCREEN_SHARE,
    MISSING_FEATURE_COUNT,
    MISSING_PATTERN_CATEGORY,
    MISSING_PATTERN_FREQUENCY,
    NOTIFICATIONS_PER_SCREEN_HOUR,
    NOTIFICATIONS_TARGET_ENCODING,
    OPENS_PER_NOTIFICATION,
    SCREENING_WINNER_FEATURES,
    SCREENING_WINNERS_NOTIFICATIONS_TE_BUNDLE,
    SCREENING_WINNERS_NOTIFICATIONS_TE_FEATURES,
    SCREENING_WINNERS_BUNDLE,
    FAMILY_CONTROL_COLUMNS,
    FEATURE_TRIAL_CONTROL_COLUMNS,
    WEEKEND_MINUS_DAILY_SCREEN,
    WORK_STUDY_SCREEN_SHARE,
    add_candidate_feature,
    add_candidate_features,
    apply_target_encoding,
    fold_frequency_values,
    fold_target_encoding_values,
    format_candidate_distribution,
    missing_pattern_keys,
    normalized_value_frequency,
    resolve_control_prediction_columns,
    safe_ratio,
    target_encoding_mapping,
)


def example_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "daily_screen_time_hours": [10.0, 0.0, np.nan],
            "social_media_hours": [4.0, 1.0, 2.0],
            "gaming_hours": [1.0, 1.0, np.nan],
            "work_study_hours": [2.0, 1.0, 3.0],
            "notifications_per_day": [100.0, 50.0, 30.0],
            "app_opens_per_day": [40.0, 20.0, np.nan],
            "weekend_screen_time": [12.0, 3.0, 4.0],
            "category": ["a", None, "b"],
            "addicted_label": [0, 1, 0],
        }
    )


class FeatureTrialTests(unittest.TestCase):
    def test_safe_ratio_preserves_missing_and_rejects_nonpositive_denominator(self) -> None:
        result = safe_ratio(
            pd.Series([10.0, 10.0, np.nan]), pd.Series([2.0, 0.0, 2.0])
        )

        self.assertEqual(result.iloc[0], 5.0)
        self.assertTrue(result.iloc[1:].isna().all())

    def test_builds_each_row_wise_candidate(self) -> None:
        expected_first_value = {
            NOTIFICATIONS_PER_SCREEN_HOUR: 10.0,
            APP_OPENS_PER_SCREEN_HOUR: 4.0,
            OPENS_PER_NOTIFICATION: 0.4,
            ENTERTAINMENT_SCREEN_SHARE: 0.5,
            WORK_STUDY_SCREEN_SHARE: 0.2,
            WEEKEND_MINUS_DAILY_SCREEN: 2.0,
            MISSING_FEATURE_COUNT: 0.0,
        }

        for candidate, expected in expected_first_value.items():
            with self.subTest(candidate=candidate):
                frame = example_frame()
                created = add_candidate_feature(frame, candidate)
                self.assertEqual(created, candidate)
                self.assertAlmostEqual(float(frame.loc[0, candidate]), expected)

    def test_missing_count_uses_original_predictors_only(self) -> None:
        frame = example_frame()

        add_candidate_feature(frame, MISSING_FEATURE_COUNT)

        self.assertEqual(frame[MISSING_FEATURE_COUNT].tolist(), [0, 1, 3])

    def test_rejects_duplicate_or_unknown_candidate(self) -> None:
        frame = example_frame()
        add_candidate_feature(frame, NOTIFICATIONS_PER_SCREEN_HOUR)
        with self.assertRaisesRegex(ValueError, "already exists"):
            add_candidate_feature(frame, NOTIFICATIONS_PER_SCREEN_HOUR)
        with self.assertRaisesRegex(ValueError, "Unknown candidate"):
            add_candidate_feature(example_frame(), "unknown")

    def test_builds_the_four_screening_winners_as_one_named_bundle(self) -> None:
        frame = example_frame()

        created = add_candidate_features(frame, SCREENING_WINNERS_BUNDLE)

        self.assertEqual(created, SCREENING_WINNER_FEATURES)
        self.assertTrue(all(feature in frame for feature in created))
        self.assertEqual(len(created), 4)

    def test_builds_winner_bundle_plus_target_encoding_placeholder(self) -> None:
        frame = example_frame()

        created = add_candidate_features(
            frame, SCREENING_WINNERS_NOTIFICATIONS_TE_BUNDLE
        )

        self.assertEqual(created, SCREENING_WINNERS_NOTIFICATIONS_TE_FEATURES)
        self.assertEqual(created[-1], NOTIFICATIONS_TARGET_ENCODING)
        self.assertTrue((frame[NOTIFICATIONS_TARGET_ENCODING] == 0.0).all())

    def test_smoothed_target_mapping_uses_mapping_rows_only(self) -> None:
        source = pd.Series(["a", "a", "b", "unseen"])
        target = pd.Series([1, 0, 1, 0])

        mapping, default = target_encoding_mapping(
            source, target, np.array([0, 1, 2]), alpha=1.0
        )
        encoded = apply_target_encoding(
            source, np.array([0, 2, 3]), mapping, default
        )

        np.testing.assert_allclose(encoded, [5 / 9, 5 / 6, 2 / 3])

    def test_target_encoding_is_inner_oof_and_outer_fold_safe(self) -> None:
        source = pd.Series(list("abcdefgh"))
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

        train_values, valid_values = fold_target_encoding_values(
            source,
            target,
            np.arange(6),
            np.array([6, 7]),
            alpha=0.0,
            inner_folds=3,
            seed=2026,
        )

        self.assertTrue(np.all((train_values > 0) & (train_values < 1)))
        np.testing.assert_allclose(valid_values, [0.5, 0.5])

    def test_resolves_family_and_feature_trial_control_predictions(self) -> None:
        family = pd.DataFrame(
            {column: [0.5] for column in FAMILY_CONTROL_COLUMNS.values()}
        )
        feature_trial = pd.DataFrame(
            {column: [0.5] for column in FEATURE_TRIAL_CONTROL_COLUMNS.values()}
        )

        self.assertEqual(
            resolve_control_prediction_columns(family), FAMILY_CONTROL_COLUMNS
        )
        self.assertEqual(
            resolve_control_prediction_columns(feature_trial),
            FEATURE_TRIAL_CONTROL_COLUMNS,
        )
        with self.assertRaisesRegex(ValueError, "supported LightGBM"):
            resolve_control_prediction_columns(pd.DataFrame({"other": [0.5]}))

    def test_normalized_frequency_includes_missing_as_a_value(self) -> None:
        result = normalized_value_frequency(pd.Series([1.0, 1.0, 2.0, np.nan]))

        np.testing.assert_allclose(result, [0.5, 0.5, 0.25, 0.25])

    def test_fold_frequency_uses_train_rows_only_and_maps_unseen_to_zero(self) -> None:
        source = pd.Series([1.0, 1.0, 2.0, 1.0, 3.0, np.nan])

        train_values, valid_values = fold_frequency_values(
            source,
            np.array([0, 1, 2]),
            np.array([3, 4, 5]),
        )

        np.testing.assert_allclose(train_values, [2 / 3, 2 / 3, 1 / 3])
        np.testing.assert_allclose(valid_values, [2 / 3, 0.0, 0.0])

    def test_builds_frequency_candidate_for_schema_and_diagnostics(self) -> None:
        frame = example_frame()

        add_candidate_feature(frame, "daily_screen_time_hours_value_frequency")

        np.testing.assert_allclose(
            frame["daily_screen_time_hours_value_frequency"],
            [1 / 3, 1 / 3, 1 / 3],
        )

    def test_missing_pattern_uses_stable_bit_positions(self) -> None:
        frame = pd.DataFrame(
            {
                "first": [1.0, np.nan, np.nan],
                "second": [1.0, 1.0, np.nan],
                "third": [np.nan, 1.0, np.nan],
            }
        )

        result = missing_pattern_keys(frame, ("first", "second", "third"))

        self.assertEqual(result.tolist(), [4, 1, 7])

    def test_missing_pattern_candidates_are_separate_features(self) -> None:
        frame = example_frame()
        feature_columns = tuple(
            column for column in frame if column not in ("id", "addicted_label")
        )
        pattern = missing_pattern_keys(frame, feature_columns)

        frequency = normalized_value_frequency(pattern)
        category = pattern.astype("string")

        self.assertEqual(len(frequency), len(frame))
        self.assertEqual(str(category.dtype), "string")
        self.assertNotEqual(MISSING_PATTERN_FREQUENCY, MISSING_PATTERN_CATEGORY)

    def test_formats_numeric_and_categorical_candidate_distributions(self) -> None:
        numeric = format_candidate_distribution(pd.Series([1.0, 2.0, 3.0]))
        categorical = format_candidate_distribution(
            pd.Series(["mask-a", "mask-a", "mask-b"], dtype="string")
        )

        self.assertIn("median=2.00000000", numeric)
        self.assertIn("mask-a:2", categorical)


if __name__ == "__main__":
    unittest.main()
