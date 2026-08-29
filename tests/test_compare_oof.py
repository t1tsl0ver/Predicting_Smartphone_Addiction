from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compare_oof import (  # noqa: E402
    load_aligned_oof,
    paired_stratified_bootstrap_deltas,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_loads_differently_named_prediction_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            control_path = directory_path / "control.csv"
            candidate_path = directory_path / "candidate.csv"
            pd.DataFrame(
                {
                    "row_index": [1, 0],
                    "target": [1, 0],
                    "control_score": [0.8, 0.2],
                }
            ).to_csv(control_path, index=False)
            pd.DataFrame(
                {
                    "row_index": [0, 1],
                    "target": [0, 1],
                    "candidate_score": [0.1, 0.9],
                }
            ).to_csv(candidate_path, index=False)

            target, control, candidate = load_aligned_oof(
                control_path,
                candidate_path,
                "unused_shared_score",
                "control_score",
                "candidate_score",
            )

        np.testing.assert_array_equal(target, np.array([0, 1], dtype=np.int8))
        np.testing.assert_allclose(control, np.array([0.2, 0.8]))
        np.testing.assert_allclose(candidate, np.array([0.1, 0.9]))

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
