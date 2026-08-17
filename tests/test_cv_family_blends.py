from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_family_blends import build_feature_variants, fold_rank_blend  # noqa: E402


class FamilyBlendTests(unittest.TestCase):
    def test_builds_exact_and_base_variants_from_accepted_features(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [1, 2],
                "age": [20.0, 30.0],
                "daily_screen_time_hours": [8.0, 10.0],
                "social_media_hours": [2.0, 3.0],
                "gaming_hours": [1.0, 2.0],
                "work_study_hours": [3.0, 3.0],
                "sleep_hours": [8.0, 7.0],
                "notifications_per_day": [50.0, 70.0],
                "app_opens_per_day": [20.0, 30.0],
                "weekend_screen_time": [9.0, 12.0],
                "gender": ["Female", "Male"],
                "stress_level": ["High", "Low"],
                "academic_work_impact": ["Yes", "No"],
                "addicted_label": [0, 1],
            }
        )

        exact, base = build_feature_variants(frame)

        self.assertEqual(len(exact), 22)
        self.assertEqual(len(base), 13)
        self.assertIn("age_exact", exact)
        self.assertNotIn("age_exact", base)
        self.assertIn("high_stress_other_screen_per_sleep", base)
        self.assertNotIn("gender", exact)
        self.assertNotIn("stress_level", exact)

    def test_blends_fold_percentile_ranks_with_fixed_exact_weight(self) -> None:
        exact = np.array([0.1, 0.4, 0.3, 0.2])
        base = np.array([0.4, 0.3, 0.2, 0.1])

        exact_rank, base_rank, blend = fold_rank_blend(exact, base, 0.6)

        np.testing.assert_allclose(exact_rank, np.array([0.25, 1.0, 0.75, 0.5]))
        np.testing.assert_allclose(base_rank, np.array([1.0, 0.75, 0.5, 0.25]))
        np.testing.assert_allclose(blend, 0.6 * exact_rank + 0.4 * base_rank)

    def test_rejects_invalid_blend_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            fold_rank_blend(np.array([0.1]), np.array([0.2]), 1.1)


if __name__ == "__main__":
    unittest.main()
