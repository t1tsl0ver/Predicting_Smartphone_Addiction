from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lightgbm import add_numeric_exact_categories  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
