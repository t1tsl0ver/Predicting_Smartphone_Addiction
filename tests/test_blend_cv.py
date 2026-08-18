from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blend_cv import load_oof, load_test, weighted_blend  # noqa: E402


class WeightedBlendTests(unittest.TestCase):
    def test_combines_predictions_with_primary_weight(self) -> None:
        primary = np.array([0.2, 0.8])
        secondary = np.array([0.6, 0.4])

        result = weighted_blend(primary, secondary, primary_weight=0.6)

        np.testing.assert_allclose(result, np.array([0.36, 0.64]))

    def test_rejects_invalid_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            weighted_blend(np.array([0.2]), np.array([0.4]), primary_weight=1.1)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            weighted_blend(
                np.array([0.2, 0.8]),
                np.array([0.4]),
                primary_weight=0.6,
            )

    def test_loaders_accept_named_prediction_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            oof_path = root / "oof.csv"
            test_path = root / "test.csv"
            pd.DataFrame(
                {
                    "row_index": [1, 0],
                    "fold": [1, 0],
                    "target": [1, 0],
                    "custom_oof": [0.8, 0.2],
                }
            ).to_csv(oof_path, index=False)
            pd.DataFrame(
                {"id": [10, 11], "custom_test": [0.3, 0.7]}
            ).to_csv(test_path, index=False)

            oof = load_oof(oof_path, "custom_oof")
            test = load_test(test_path, "custom_test")

            self.assertEqual(oof["row_index"].tolist(), [0, 1])
            self.assertEqual(oof["lightgbm_fold_rank"].tolist(), [0.2, 0.8])
            self.assertEqual(test["lightgbm_cv_rank"].tolist(), [0.3, 0.7])


if __name__ == "__main__":
    unittest.main()
