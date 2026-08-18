from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from compare_oof import DEFAULT_TEST_SIZE, paired_stratified_bootstrap_deltas
from progress_logging import RunLogger


OOF_PREDICTION = "lightgbm_fold_rank"
TEST_PREDICTION = "lightgbm_cv_rank"
BLEND_PREDICTION = "rank_blend_prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend aligned LightGBM CV rank predictions and create a submission."
    )
    parser.add_argument("--primary-oof", type=Path, required=True)
    parser.add_argument("--primary-test", type=Path, required=True)
    parser.add_argument("--primary-oof-column", type=str, default=OOF_PREDICTION)
    parser.add_argument("--primary-test-column", type=str, default=TEST_PREDICTION)
    parser.add_argument("--primary-name", type=str, required=True)
    parser.add_argument("--primary-weight", type=float, required=True)
    parser.add_argument("--secondary-oof", type=Path, required=True)
    parser.add_argument("--secondary-test", type=Path, required=True)
    parser.add_argument("--secondary-oof-column", type=str, default=OOF_PREDICTION)
    parser.add_argument("--secondary-test-column", type=str, default=TEST_PREDICTION)
    parser.add_argument("--secondary-name", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--bootstrap-sample-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def weighted_blend(
    primary: np.ndarray,
    secondary: np.ndarray,
    primary_weight: float,
) -> np.ndarray:
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError("Primary weight must be between 0 and 1")
    primary = np.asarray(primary, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    if primary.shape != secondary.shape:
        raise ValueError("Prediction arrays must have identical shapes")
    if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
        raise ValueError("Prediction arrays must be finite")
    return primary_weight * primary + (1.0 - primary_weight) * secondary


def load_oof(path: Path, prediction_column: str = OOF_PREDICTION) -> pd.DataFrame:
    columns = ["row_index", "fold", "target", prediction_column]
    frame = pd.read_csv(path, usecols=columns).sort_values("row_index")
    if frame["row_index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate OOF row indices")
    return frame.rename(columns={prediction_column: OOF_PREDICTION}).reset_index(
        drop=True
    )


def load_test(path: Path, prediction_column: str = TEST_PREDICTION) -> pd.DataFrame:
    columns = ["id", prediction_column]
    frame = pd.read_csv(path, usecols=columns)
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate test ids")
    return frame.rename(columns={prediction_column: TEST_PREDICTION})


def main() -> None:
    args = parse_args()
    for label in (args.run_id, args.primary_name, args.secondary_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
            raise ValueError(f"Unsafe label: {label!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("blend-cv", args.output_dir / "logs", args.run_id)

    with logger.step("Load and align OOF predictions"):
        primary_oof = load_oof(args.primary_oof, args.primary_oof_column)
        secondary_oof = load_oof(args.secondary_oof, args.secondary_oof_column)
        for column in ("row_index", "fold", "target"):
            if not primary_oof[column].equals(secondary_oof[column]):
                raise ValueError(f"Primary and secondary OOF {column} values differ")

    with logger.step("Load and align test predictions"):
        primary_test = load_test(args.primary_test, args.primary_test_column)
        secondary_test = load_test(args.secondary_test, args.secondary_test_column)
        sample = pd.read_csv(args.sample_submission)
        if list(sample.columns) != ["id", "addicted_label"]:
            raise ValueError("Unexpected sample submission schema")
        if not primary_test["id"].equals(secondary_test["id"]):
            raise ValueError("Primary and secondary test ids differ")
        if not primary_test["id"].equals(sample["id"]):
            raise ValueError("Prediction and sample submission ids differ")

    target = primary_oof["target"].to_numpy(dtype=np.int8)
    primary_oof_prediction = primary_oof[OOF_PREDICTION].to_numpy(dtype=np.float64)
    secondary_oof_prediction = secondary_oof[OOF_PREDICTION].to_numpy(
        dtype=np.float64
    )
    blend_oof_prediction = weighted_blend(
        primary_oof_prediction,
        secondary_oof_prediction,
        args.primary_weight,
    )
    blend_test_prediction = weighted_blend(
        primary_test[TEST_PREDICTION].to_numpy(dtype=np.float64),
        secondary_test[TEST_PREDICTION].to_numpy(dtype=np.float64),
        args.primary_weight,
    )

    primary_auc = roc_auc_score(target, primary_oof_prediction)
    secondary_auc = roc_auc_score(target, secondary_oof_prediction)
    blend_auc = roc_auc_score(target, blend_oof_prediction)
    logger.info(
        f"OOF | {args.primary_name}={primary_auc:.10f}, "
        f"{args.secondary_name}={secondary_auc:.10f}, blend={blend_auc:.10f}, "
        f"delta_vs_{args.primary_name}={blend_auc - primary_auc:+.10f}"
    )

    fold_results: list[dict[str, float | int]] = []
    for fold in sorted(primary_oof["fold"].unique()):
        mask = primary_oof["fold"].to_numpy() == fold
        fold_primary_auc = roc_auc_score(target[mask], primary_oof_prediction[mask])
        fold_secondary_auc = roc_auc_score(
            target[mask], secondary_oof_prediction[mask]
        )
        fold_blend_auc = roc_auc_score(target[mask], blend_oof_prediction[mask])
        fold_results.append(
            {
                "fold": int(fold),
                "primary_auc": fold_primary_auc,
                "secondary_auc": fold_secondary_auc,
                "blend_auc": fold_blend_auc,
                "delta_vs_primary": fold_blend_auc - fold_primary_auc,
            }
        )
        logger.info(
            f"FOLD {fold} | blend={fold_blend_auc:.10f}, "
            f"delta_vs_{args.primary_name}={fold_blend_auc - fold_primary_auc:+.10f}"
        )

    with logger.step("Paired stratified bootstrap against primary"):
        deltas = paired_stratified_bootstrap_deltas(
            target,
            primary_oof_prediction,
            blend_oof_prediction,
            args.bootstrap_repeats,
            args.bootstrap_sample_size,
            args.bootstrap_seed,
        )

    metrics = {
        "run_id": args.run_id,
        "method": "weighted_rank_blend",
        "primary": {
            "name": args.primary_name,
            "oof_path": str(args.primary_oof),
            "test_path": str(args.primary_test),
            "oof_column": args.primary_oof_column,
            "test_column": args.primary_test_column,
            "weight": args.primary_weight,
            "oof_auc": primary_auc,
        },
        "secondary": {
            "name": args.secondary_name,
            "oof_path": str(args.secondary_oof),
            "test_path": str(args.secondary_test),
            "oof_column": args.secondary_oof_column,
            "test_column": args.secondary_test_column,
            "weight": 1.0 - args.primary_weight,
            "oof_auc": secondary_auc,
        },
        "blend_oof_auc": blend_auc,
        "auc_delta_vs_primary": blend_auc - primary_auc,
        "prediction_pearson_correlation": float(
            np.corrcoef(primary_oof_prediction, secondary_oof_prediction)[0, 1]
        ),
        "prediction_spearman_correlation": float(
            spearmanr(primary_oof_prediction, secondary_oof_prediction).statistic
        ),
        "fold_results": fold_results,
        "paired_stratified_bootstrap_vs_primary": {
            "scope": "row-sampling uncertainty only; not training or LB-split uncertainty",
            "repeats": args.bootstrap_repeats,
            "sample_size": args.bootstrap_sample_size,
            "seed": args.bootstrap_seed,
            "mean_auc_delta": float(deltas.mean()),
            "standard_deviation": float(deltas.std(ddof=1)),
            "ci95_lower": float(np.quantile(deltas, 0.025)),
            "ci95_upper": float(np.quantile(deltas, 0.975)),
            "probability_delta_positive": float(np.mean(deltas > 0)),
        },
    }

    metrics_path = args.output_dir / f"metrics_{args.run_id}_rank_blend.json"
    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    test_path = args.output_dir / f"test_predictions_{args.run_id}.csv"
    with logger.step("Save blend metrics and predictions"):
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "row_index": primary_oof["row_index"],
                "fold": primary_oof["fold"],
                "target": target,
                BLEND_PREDICTION: blend_oof_prediction,
            }
        ).to_csv(oof_path, index=False)
        pd.DataFrame(
            {
                "id": primary_test["id"],
                BLEND_PREDICTION: blend_test_prediction,
            }
        ).to_csv(test_path, index=False)

        if args.submission is not None:
            submission = sample.copy()
            submission["addicted_label"] = blend_test_prediction
            submission.to_csv(args.submission, index=False)
            digest = hashlib.sha256(args.submission.read_bytes()).hexdigest()
            metadata = {
                "run_id": args.run_id,
                "selected_prediction": "weighted_rank_blend",
                "primary_name": args.primary_name,
                "primary_weight": args.primary_weight,
                "secondary_name": args.secondary_name,
                "secondary_weight": 1.0 - args.primary_weight,
                "rows": len(submission),
                "minimum_prediction": float(blend_test_prediction.min()),
                "maximum_prediction": float(blend_test_prediction.max()),
                "sha256": digest,
            }
            args.submission.with_suffix(".metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info(f"Submission saved to {args.submission}")
            logger.info(f"SHA-256={digest}")

    logger.finish()


if __name__ == "__main__":
    main()
