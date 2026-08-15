from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lightgbm import add_numeric_exact_categories, make_model  # noqa: E402


class ExactValueCategoryTests(unittest.TestCase):
    def test_preserves_exact_values_and_missingness_as_strings(self) -> None:
        frame = pd.DataFrame({"value": [1.25, 1.25, 2.5, None]})

        created = add_numeric_exact_categories(frame, ["value"])

        self.assertEqual(created, ["value_exact"])
        self.assertEqual(
            frame["value_exact"].tolist(),
            ["1.25", "1.25", "2.5", pd.NA],
        )
        self.assertEqual(frame["value"].tolist()[:3], [1.25, 1.25, 2.5])

    def test_rejects_missing_source_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            add_numeric_exact_categories(pd.DataFrame({"value": [1]}), ["other"])

    def test_forwards_categorical_smoothing_to_lightgbm(self) -> None:
        args = SimpleNamespace(
            iterations=500,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=0.0,
            cat_smooth=20.0,
        )

        model = make_model(args, fold_seed=2027)

        self.assertEqual(model.get_params()["cat_smooth"], 20.0)


if __name__ == "__main__":
    unittest.main()
