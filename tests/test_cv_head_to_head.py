from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_head_to_head import (  # noqa: E402
    HIGH_STRESS_OTHER_SCREEN_PER_SLEEP,
    OTHER_SCREEN_TIME,
    SCREEN_SOCIAL_LOAD,
    add_accepted_features,
    catboost_caught_up,
)


class HeadToHeadFeatureTests(unittest.TestCase):
    def test_builds_the_accepted_feature_set_before_raw_stress_is_dropped(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [20.0, 21.0],
                "daily_screen_time_hours": [10.0, 8.0],
                "social_media_hours": [3.0, 2.0],
                "gaming_hours": [2.0, 1.0],
                "work_study_hours": [1.0, 2.0],
                "sleep_hours": [8.0, 5.0],
                "notifications_per_day": [100.0, 80.0],
                "app_opens_per_day": [40.0, 30.0],
                "weekend_screen_time": [12.0, 9.0],
                "stress_level": ["High", "Low"],
            }
        )

        exact_features = add_accepted_features(frame)

        self.assertEqual(len(exact_features), 9)
        self.assertEqual(frame[OTHER_SCREEN_TIME].tolist(), [4.0, 3.0])
        self.assertEqual(frame[SCREEN_SOCIAL_LOAD].tolist(), [30.0, 16.0])
        self.assertEqual(
            frame[HIGH_STRESS_OTHER_SCREEN_PER_SLEEP].tolist(), [0.5, 0.0]
        )
        self.assertEqual(
            frame["daily_screen_time_hours_exact"].tolist(), ["10.0", "8.0"]
        )


class HeadToHeadDecisionTests(unittest.TestCase):
    def test_accepts_catboost_at_exact_deficit_boundary(self) -> None:
        self.assertTrue(catboost_caught_up(0.96420, 0.96430, tolerance=0.00010))

    def test_rejects_catboost_outside_deficit_boundary(self) -> None:
        self.assertFalse(catboost_caught_up(0.96419, 0.96430, tolerance=0.00010))

    def test_accepts_catboost_when_it_wins(self) -> None:
        self.assertTrue(catboost_caught_up(0.96431, 0.96430, tolerance=0.00010))

    def test_rejects_negative_tolerance(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            catboost_caught_up(0.9, 0.9, tolerance=-0.1)


if __name__ == "__main__":
    unittest.main()
