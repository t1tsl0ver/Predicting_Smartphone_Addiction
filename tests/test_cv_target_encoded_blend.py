from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_target_encoded_blend import frequency_values_for_test  # noqa: E402


class TargetEncodedBlendTests(unittest.TestCase):
    def test_test_frequency_uses_outer_train_only_and_unseen_is_zero(self) -> None:
        train = pd.Series([1.0, 1.0, 2.0, 2.0, 2.0])
        test = pd.Series([1.0, 2.0, 3.0, np.nan])

        values = frequency_values_for_test(train, test, np.array([0, 1, 2]))

        np.testing.assert_allclose(values, [2 / 3, 1 / 3, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
