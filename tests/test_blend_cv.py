from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blend_cv import weighted_blend  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
