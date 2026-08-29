from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_feature_trial import (  # noqa: E402
    NUMERIC_VALUE_MEAN_SURPRISAL,
    TARGET_ENCODING_CANDIDATES,
    TRIAL_065_WITHOUT_NUMERIC_SURPRISAL_BUNDLE,
    TRIAL_065_WITHOUT_NUMERIC_SURPRISAL_FEATURES,
)
from cv_target_encoded_blend import (  # noqa: E402
    FEATURE_BUNDLE_CANDIDATES,
    TRIAL_066_NO_TE_BUNDLE_MODE,
    build_feature_frames,
    frequency_values_for_test,
)


class TargetEncodedBlendTests(unittest.TestCase):
    def test_test_frequency_uses_outer_train_only_and_unseen_is_zero(self) -> None:
        train = pd.Series([1.0, 1.0, 2.0, 2.0, 2.0])
        test = pd.Series([1.0, 2.0, 3.0, np.nan])

        values = frequency_values_for_test(train, test, np.array([0, 1, 2]))

        np.testing.assert_allclose(values, [2 / 3, 1 / 3, 0.0, 0.0])

    def test_builds_full_trial_066_bundle_without_target_encoding(self) -> None:
        train = example_frame(include_target=True)
        test = example_frame(include_target=False)

        candidates, exact_features, base_features, variants = build_feature_frames(
            train,
            test,
            FEATURE_BUNDLE_CANDIDATES[TRIAL_066_NO_TE_BUNDLE_MODE],
        )

        self.assertEqual(candidates, TRIAL_065_WITHOUT_NUMERIC_SURPRISAL_FEATURES)
        self.assertNotIn(NUMERIC_VALUE_MEAN_SURPRISAL, candidates)
        self.assertFalse(
            any(feature in TARGET_ENCODING_CANDIDATES for feature in candidates)
        )
        self.assertEqual(variants["exact"][3], exact_features)
        self.assertEqual(variants["base"][3], base_features)
        self.assertGreater(len(exact_features), len(base_features))


def example_frame(include_target: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [20.0, 21.0, 22.0],
            "daily_screen_time_hours": [8.0, 9.0, 10.0],
            "social_media_hours": [2.0, 3.0, 4.0],
            "gaming_hours": [1.0, 1.5, 2.0],
            "work_study_hours": [3.0, 2.0, 2.0],
            "sleep_hours": [7.0, 6.5, 6.0],
            "notifications_per_day": [100.0, 120.0, 140.0],
            "app_opens_per_day": [80.0, 90.0, 100.0],
            "weekend_screen_time": [9.0, 10.0, 11.0],
            "gender": ["Female", "Male", "Other"],
            "stress_level": ["Low", "Medium", "High"],
            "academic_work_impact": ["No", "Yes", "No"],
        }
    )
    if include_target:
        frame["addicted_label"] = [0, 1, 1]
    return frame


if __name__ == "__main__":
    unittest.main()
