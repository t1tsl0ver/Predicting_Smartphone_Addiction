from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_drop_exact_feature import (  # noqa: E402
    expected_fold_numbers,
    remove_exact_feature,
)
from trial_config import TRIAL_FOLDS  # noqa: E402


class DropExactFeatureTests(unittest.TestCase):
    def test_removes_exactly_the_requested_feature(self) -> None:
        features = ["age", "age_exact", "gaming_hours_exact"]

        candidate = remove_exact_feature(features, "gaming_hours_exact")

        self.assertEqual(candidate, ["age", "age_exact"])
        self.assertEqual(features, ["age", "age_exact", "gaming_hours_exact"])

    def test_rejects_non_exact_or_missing_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an _exact"):
            remove_exact_feature(["gaming_hours"], "gaming_hours")
        with self.assertRaisesRegex(ValueError, "missing feature"):
            remove_exact_feature(["age_exact"], "gaming_hours_exact")

    def test_trial_default_is_permanently_two_folds(self) -> None:
        self.assertEqual(TRIAL_FOLDS, 2)

    def test_expected_fold_numbers_cover_every_row_once(self) -> None:
        target = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        first = expected_fold_numbers(target, folds=2, seed=2026)
        second = expected_fold_numbers(target, folds=2, seed=2026)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(first), {1, 2})
        self.assertEqual(np.bincount(first)[1:].tolist(), [4, 4])


if __name__ == "__main__":
    unittest.main()
