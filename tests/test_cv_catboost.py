from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_catboost import (  # noqa: E402
    DROPPED_EXACT_FEATURE,
    build_catboost_features,
    load_fold_checkpoint,
    make_full_catboost,
    save_fold_checkpoint,
)


def model_args() -> argparse.Namespace:
    return argparse.Namespace(
        iterations=3000,
        learning_rate=0.01,
        depth=8,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        border_count=254,
        max_ctr_complexity=1,
        leaf_estimation_iterations=10,
        subsample=0.8,
        snapshot_interval=300,
    )


class FullCatBoostConfigurationTests(unittest.TestCase):
    def test_uses_the_full_low_learning_rate_catboost_configuration(self) -> None:
        model = make_full_catboost(model_args(), fold_seed=2027)
        parameters = model.get_params()

        self.assertEqual(parameters["iterations"], 3000)
        self.assertEqual(parameters["learning_rate"], 0.01)
        self.assertEqual(parameters["depth"], 8)
        self.assertEqual(parameters["l2_leaf_reg"], 5.0)
        self.assertEqual(parameters["bootstrap_type"], "MVS")
        self.assertEqual(parameters["subsample"], 0.8)
        self.assertEqual(parameters["eval_metric"], "AUC")
        self.assertFalse(parameters["allow_writing_files"])

    def test_removes_only_the_accepted_gaming_exact_copy(self) -> None:
        frame = pd.DataFrame(
            {
                "id": [0, 1],
                "age": [20.0, 21.0],
                "daily_screen_time_hours": [10.0, 8.0],
                "social_media_hours": [3.0, 2.0],
                "gaming_hours": [2.0, 1.0],
                "work_study_hours": [1.0, 2.0],
                "sleep_hours": [8.0, 5.0],
                "notifications_per_day": [100.0, 80.0],
                "app_opens_per_day": [40.0, 30.0],
                "weekend_screen_time": [12.0, 9.0],
                "gender": ["Male", "Female"],
                "stress_level": ["High", "Low"],
                "academic_work_impact": ["Positive", "Negative"],
                "addicted_label": [1, 0],
            }
        )

        features = build_catboost_features(frame)

        self.assertNotIn(DROPPED_EXACT_FEATURE, features)
        self.assertIn("gaming_hours", features)
        self.assertNotIn("gender", features)
        self.assertNotIn("stress_level", features)
        self.assertEqual(len(features), 21)


class FoldCheckpointTests(unittest.TestCase):
    def test_round_trips_a_valid_fold_checkpoint(self) -> None:
        result = {
            "signature": "abc123",
            "fold": 1,
            "valid_indices": np.array([1, 3]),
            "valid_probability": np.array([0.2, 0.8]),
            "test_probability": np.array([0.4, 0.6, 0.5]),
            "feature_importance": np.array([1.0, 2.0]),
            "best_iteration": 120,
            "trained_iterations": 520,
            "tree_count": 120,
            "validation_auc": 0.9,
            "elapsed_seconds": 12.5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold-1.npz"
            save_fold_checkpoint(path, result)
            loaded = load_fold_checkpoint(
                path,
                signature="abc123",
                fold=1,
                valid_indices=np.array([1, 3]),
                test_count=3,
                feature_count=2,
            )

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded["best_iteration"], 120)
        np.testing.assert_array_equal(loaded["test_probability"], [0.4, 0.6, 0.5])

    def test_rejects_a_checkpoint_from_another_configuration(self) -> None:
        result = {
            "signature": "old",
            "fold": 1,
            "valid_indices": np.array([1]),
            "valid_probability": np.array([0.2]),
            "test_probability": np.array([0.4]),
            "feature_importance": np.array([1.0]),
            "best_iteration": 2,
            "trained_iterations": 3,
            "tree_count": 2,
            "validation_auc": 0.9,
            "elapsed_seconds": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fold-1.npz"
            save_fold_checkpoint(path, result)
            loaded = load_fold_checkpoint(
                path,
                signature="new",
                fold=1,
                valid_indices=np.array([1]),
                test_count=1,
                feature_count=1,
            )

        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
