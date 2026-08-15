from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from progress_logging import RunLogger


TARGET = "addicted_label"
ID_COLUMN = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the CV-selected Kaggle submission."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/metrics_trial-007_cv.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/test_predictions_trial-007.csv"),
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("artifacts/logs"),
        help="Directory for persistent execution logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = RunLogger(
        "create-cv-submission",
        args.log_dir,
        args.metrics.stem,
    )
    with logger.step("Load metrics, predictions and sample submission"):
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        predictions = pd.read_csv(args.predictions)
        sample = pd.read_csv(args.sample_submission)
    selected = metrics["selected_submission"]
    prediction_column = {
        "lightgbm": "lightgbm_cv_rank",
        "blend": "blend_cv_rank",
    }.get(selected)
    with logger.step("Validate selected prediction column"):
        if prediction_column is None:
            raise ValueError(f"Unknown selected submission: {selected!r}")
        if not predictions[ID_COLUMN].equals(sample[ID_COLUMN]):
            raise ValueError("Prediction and sample submission ids differ")

        values = predictions[prediction_column].to_numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Submission predictions must be finite and within [0, 1]")
    logger.info(
        f"Selected candidate={selected}; column={prediction_column}; rows={len(values)}"
    )

    with logger.step("Write submission and metadata"):
        submission = sample.copy()
        submission[TARGET] = values
        submission.to_csv(args.output, index=False)

        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
        metadata = {
            "selected_submission": selected,
            "prediction_column": prediction_column,
            "rows": len(submission),
            "minimum_prediction": float(values.min()),
            "maximum_prediction": float(values.max()),
            "sha256": digest,
        }
        metadata_path = args.output.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    logger.info(f"Submission saved to {args.output} ({len(submission)} rows)")
    logger.info(f"Metadata saved to {metadata_path}")
    logger.info(f"Prediction range=[{values.min():.10f}, {values.max():.10f}]")
    logger.info(f"SHA-256={digest}")
    logger.finish()


if __name__ == "__main__":
    main()
