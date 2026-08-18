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
    NOTIFICATIONS_PER_SCREEN_HOUR,
    OPENS_PER_NOTIFICATION,
    WEEKEND_MINUS_DAILY_SCREEN,
    WORK_STUDY_SCREEN_SHARE,
    add_candidate_feature,
    safe_ratio,
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


if __name__ == "__main__":
    unittest.main()
