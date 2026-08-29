from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


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
    load_cached_exact_predictions,
    resolve_variant_training_configs,
)


class TargetEncodedBlendTests(unittest.TestCase):
    def test_resolves_independent_variant_training_configs(self) -> None:
        args = SimpleNamespace(
            iterations=3000,
            learning_rate=0.01,
            early_stopping_rounds=400,
            exact_iterations=None,
            base_iterations=8000,
            exact_learning_rate=None,
            base_learning_rate=None,
            exact_early_stopping_rounds=None,
            base_early_stopping_rounds=800,
        )

        configs = resolve_variant_training_configs(args)

        self.assertEqual(
            configs["exact"],
            {
                "iterations": 3000,
                "learning_rate": 0.01,
                "early_stopping_rounds": 400,
            },
        )
        self.assertEqual(
            configs["base"],
            {
                "iterations": 8000,
                "learning_rate": 0.01,
                "early_stopping_rounds": 800,
            },
        )

    def test_loads_and_validates_cached_exact_predictions(self) -> None:
        target = pd.Series([0, 0, 0, 1, 1, 1], name="addicted_label")
        row_index = np.arange(len(target))
        expected_fold = np.zeros(len(target), dtype=np.int8)
        splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
        for fold, (_, valid_indices) in enumerate(
            splitter.split(row_index, target), start=1
        ):
            expected_fold[valid_indices] = fold

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            oof_path = directory_path / "oof.csv"
            test_path = directory_path / "test.csv"
            pd.DataFrame(
                {
                    "row_index": row_index,
                    "fold": expected_fold,
                    "target": target,
                    "exact_probability": np.linspace(0.1, 0.9, len(target)),
                    "exact_fold_rank": np.linspace(0.2, 1.0, len(target)),
                }
            ).sample(frac=1.0, random_state=3).to_csv(oof_path, index=False)
            pd.DataFrame(
                {
                    "id": [10, 11],
                    "exact_probability": [0.25, 0.75],
                    "exact_cv_rank": [0.4, 0.8],
                }
            ).to_csv(test_path, index=False)

            cached = load_cached_exact_predictions(
                oof_path,
                test_path,
                target,
                pd.Series([10, 11], name="id"),
                folds=2,
                seed=7,
            )

        np.testing.assert_array_equal(cached["fold"], expected_fold)
        np.testing.assert_allclose(cached["oof_probability"], np.linspace(0.1, 0.9, 6))
        np.testing.assert_allclose(cached["test_rank"], [0.4, 0.8])

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
