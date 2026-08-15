from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


DEFAULT_TEST_SIZE = 296_302


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two aligned OOF prediction files with a paired stratified "
            "bootstrap. The interval measures row-sampling uncertainty only."
        )
    )
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--prediction-column",
        default="lightgbm_probability",
    )
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_aligned_oof(
    control_path: Path,
    candidate_path: Path,
    prediction_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ["row_index", "target", prediction_column]
    control = pd.read_csv(control_path, usecols=required).sort_values("row_index")
    candidate = pd.read_csv(candidate_path, usecols=required).sort_values("row_index")

    if control["row_index"].duplicated().any() or candidate["row_index"].duplicated().any():
        raise ValueError("OOF row_index values must be unique")
    if not control["row_index"].reset_index(drop=True).equals(
        candidate["row_index"].reset_index(drop=True)
    ):
        raise ValueError("Control and candidate row_index values differ")
    if not control["target"].reset_index(drop=True).equals(
        candidate["target"].reset_index(drop=True)
    ):
        raise ValueError("Control and candidate targets differ")

    target = control["target"].to_numpy(dtype=np.int8)
    control_prediction = control[prediction_column].to_numpy(dtype=np.float64)
    candidate_prediction = candidate[prediction_column].to_numpy(dtype=np.float64)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("Target must contain exactly the binary values 0 and 1")
    if not np.isfinite(control_prediction).all() or not np.isfinite(
        candidate_prediction
    ).all():
        raise ValueError("Predictions must be finite")
    return target, control_prediction, candidate_prediction


def paired_stratified_bootstrap_deltas(
    target: np.ndarray,
    control_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    repeats: int,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    if repeats < 2:
        raise ValueError("Bootstrap repeats must be at least 2")
    if sample_size < 2:
        raise ValueError("Bootstrap sample size must be at least 2")

    positive_indices = np.flatnonzero(target == 1)
    negative_indices = np.flatnonzero(target == 0)
    positive_count = int(round(sample_size * len(positive_indices) / len(target)))
    positive_count = min(max(positive_count, 1), sample_size - 1)
    negative_count = sample_size - positive_count
    sample_target = np.concatenate(
        [
            np.ones(positive_count, dtype=np.int8),
            np.zeros(negative_count, dtype=np.int8),
        ]
    )

    rng = np.random.default_rng(seed)
    deltas = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled_indices = np.concatenate(
            [
                rng.choice(positive_indices, size=positive_count, replace=True),
                rng.choice(negative_indices, size=negative_count, replace=True),
            ]
        )
        control_auc = roc_auc_score(
            sample_target, control_prediction[sampled_indices]
        )
        candidate_auc = roc_auc_score(
            sample_target, candidate_prediction[sampled_indices]
        )
        deltas[repeat] = candidate_auc - control_auc
    return deltas


def main() -> None:
    args = parse_args()
    target, control_prediction, candidate_prediction = load_aligned_oof(
        args.control,
        args.candidate,
        args.prediction_column,
    )
    control_auc = roc_auc_score(target, control_prediction)
    candidate_auc = roc_auc_score(target, candidate_prediction)
    deltas = paired_stratified_bootstrap_deltas(
        target,
        control_prediction,
        candidate_prediction,
        args.repeats,
        args.sample_size,
        args.seed,
    )
    result = {
        "control": str(args.control),
        "candidate": str(args.candidate),
        "prediction_column": args.prediction_column,
        "rows": len(target),
        "control_auc": control_auc,
        "candidate_auc": candidate_auc,
        "auc_delta": candidate_auc - control_auc,
        "prediction_pearson_correlation": float(
            np.corrcoef(control_prediction, candidate_prediction)[0, 1]
        ),
        "prediction_spearman_correlation": float(
            spearmanr(control_prediction, candidate_prediction).statistic
        ),
        "paired_stratified_bootstrap": {
            "scope": "row-sampling uncertainty only; not training or LB-split uncertainty",
            "repeats": args.repeats,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "mean_auc_delta": float(deltas.mean()),
            "standard_deviation": float(deltas.std(ddof=1)),
            "ci95_lower": float(np.quantile(deltas, 0.025)),
            "ci95_upper": float(np.quantile(deltas, 0.975)),
            "probability_delta_positive": float(np.mean(deltas > 0)),
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
