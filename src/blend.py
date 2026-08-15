from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from progress_logging import RunLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed rank blends of two validation predictions."
    )
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--primary-name", type=str, required=True)
    parser.add_argument("--secondary-name", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def load_predictions(path: Path, prediction_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = ["row_index", "target", "prediction"]
    if list(frame.columns) != expected:
        raise ValueError(f"{path} columns must be {expected}")
    if frame["row_index"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row indices")
    if frame["prediction"].isna().any():
        raise ValueError(f"{path} contains missing predictions")
    return frame.rename(columns={"prediction": prediction_name})


def main() -> None:
    args = parse_args()
    for label in (args.run_id, args.primary_name, args.secondary_name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label):
            raise ValueError(f"Unsafe label: {label!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("blend", args.output_dir / "logs", args.run_id)

    with logger.step("Load and align validation predictions"):
        primary = load_predictions(args.primary, "primary_prediction")
        secondary = load_predictions(args.secondary, "secondary_prediction")
        merged = primary.merge(
            secondary,
            on="row_index",
            how="inner",
            validate="one_to_one",
            suffixes=("_primary", "_secondary"),
        )
        if len(merged) != len(primary) or len(merged) != len(secondary):
            raise ValueError("Prediction files contain different validation rows")
        if not merged["target_primary"].equals(merged["target_secondary"]):
            raise ValueError("Prediction files contain different targets")
    logger.info(
        f"Predictions aligned | rows={len(merged)}, "
        f"primary={args.primary_name}, secondary={args.secondary_name}"
    )

    y = merged["target_primary"]
    primary_prediction = merged["primary_prediction"]
    secondary_prediction = merged["secondary_prediction"]
    primary_rank = primary_prediction.rank(method="average", pct=True)
    secondary_rank = secondary_prediction.rank(method="average", pct=True)

    results: dict[str, object] = {
        "run_id": args.run_id,
        "primary": args.primary_name,
        "secondary": args.secondary_name,
        "row_count": len(merged),
        "primary_auc": roc_auc_score(y, primary_prediction),
        "secondary_auc": roc_auc_score(y, secondary_prediction),
        "prediction_pearson": primary_prediction.corr(secondary_prediction),
        "prediction_spearman": primary_prediction.corr(
            secondary_prediction, method="spearman"
        ),
        "rank_blends": {},
    }

    logger.info(f"{args.primary_name} AUC={results['primary_auc']:.10f}")
    logger.info(f"{args.secondary_name} AUC={results['secondary_auc']:.10f}")
    logger.info(f"Prediction Pearson={results['prediction_pearson']:.6f}")
    logger.info(f"Prediction Spearman={results['prediction_spearman']:.6f}")

    blend_predictions: dict[str, pd.Series] = {}
    with logger.step("Evaluate fixed rank blends"):
        for primary_weight in (0.15, 0.30, 0.50, 0.70, 0.85):
            secondary_weight = 1.0 - primary_weight
            label = f"{int(primary_weight * 100)}_{int(secondary_weight * 100)}"
            prediction = (
                primary_weight * primary_rank + secondary_weight * secondary_rank
            )
            auc = roc_auc_score(y, prediction)
            results["rank_blends"][label] = {
                "primary_weight": primary_weight,
                "secondary_weight": secondary_weight,
                "validation_auc": auc,
                "gain_vs_primary": auc - results["primary_auc"],
            }
            blend_predictions[label] = prediction
            logger.info(
                f"Rank blend {label} AUC={auc:.10f} "
                f"(vs primary {auc - results['primary_auc']:+.10f})"
            )

    best_label = max(
        results["rank_blends"],
        key=lambda label: results["rank_blends"][label]["validation_auc"],
    )
    results["best_blend"] = best_label
    logger.info(f"Best blend={best_label}")
    with logger.step("Save blend metrics and predictions"):
        (args.output_dir / f"metrics_{args.run_id}_blend.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "row_index": merged["row_index"],
                "target": y,
                "prediction": blend_predictions[best_label],
            }
        ).to_csv(
            args.output_dir / f"validation_predictions_{args.run_id}_best_blend.csv",
            index=False,
        )
    logger.finish()


if __name__ == "__main__":
    main()
