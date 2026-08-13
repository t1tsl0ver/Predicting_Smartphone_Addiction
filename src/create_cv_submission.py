from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.predictions)
    sample = pd.read_csv(args.sample_submission)
    selected = metrics["selected_submission"]
    prediction_column = {
        "lightgbm": "lightgbm_cv_rank",
        "blend": "blend_cv_rank",
    }.get(selected)
    if prediction_column is None:
        raise ValueError(f"Unknown selected submission: {selected!r}")
    if not predictions[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Prediction and sample submission ids differ")

    values = predictions[prediction_column].to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Submission predictions must be finite and within [0, 1]")
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
    print(f"Selected candidate: {selected}")
    print(f"Submission saved to {args.output} ({len(submission)} rows)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
