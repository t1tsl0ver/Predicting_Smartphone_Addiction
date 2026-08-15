from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compare_oof import paired_stratified_bootstrap_deltas  # noqa: E402


class PairedBootstrapTests(unittest.TestCase):
    def test_detects_consistently_better_candidate(self) -> None:
        target = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
        control = np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.9])
        candidate = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        deltas = paired_stratified_bootstrap_deltas(
            target,
            control,
            candidate,
            repeats=20,
            sample_size=6,
            seed=7,
        )

        self.assertTrue(np.all(deltas >= 0))
        self.assertGreater(deltas.mean(), 0)

    def test_validates_bootstrap_budget(self) -> None:
        target = np.array([0, 1], dtype=np.int8)
        prediction = np.array([0.1, 0.9])

        with self.assertRaisesRegex(ValueError, "at least 2"):
            paired_stratified_bootstrap_deltas(
                target,
                prediction,
                prediction,
                repeats=1,
                sample_size=2,
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
