from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from blend_cv import weighted_blend
from cv_family_blends import build_feature_variants
from cv_head_to_head import make_lightgbm
from cv_lightgbm import ID_COLUMN, TARGET, percentile_rank, prepare_lightgbm
from progress_logging import LightGBMProgressCallback, RunLogger, format_duration
from trial_config import (
    TRIAL_EXACT_WEIGHT,
    TRIAL_ITERATIONS,
    TRIAL_LIGHTGBM_LEARNING_RATE,
    TRIAL_SEED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable lightweight LightGBM exact/base OOF control."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=TRIAL_SEED)
    parser.add_argument("--iterations", type=int, default=TRIAL_ITERATIONS)
    parser.add_argument(
        "--learning-rate", type=float, default=TRIAL_LIGHTGBM_LEARNING_RATE
    )
    parser.add_argument("--exact-weight", type=float, default=TRIAL_EXACT_WEIGHT)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", default="lightweight-5f-control")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if not 0 <= args.exact_weight <= 1:
        raise ValueError("--exact-weight must be between 0 and 1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("Invalid --run-id")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("cv-lightweight-control", args.output_dir / "logs", args.run_id)
    with logger.step("Load train and build feature variants"):
        train = pd.read_csv(args.train)
        if TARGET not in train or ID_COLUMN not in train:
            raise ValueError("Unexpected training schema")
        exact_features, base_features = build_feature_variants(train)
        y = train[TARGET]
        exact_categorical = train[exact_features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        base_categorical = train[base_features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        empty = train[exact_features].iloc[:0].copy()
        x_exact, _ = prepare_lightgbm(
            train[exact_features], empty, exact_categorical
        )
        x_base = x_exact[base_features]

    variants = {
        "exact": (x_exact, exact_categorical, exact_features),
        "base": (x_base, base_categorical, base_features),
    }
    row_count = len(train)
    probabilities = {
        variant: np.full(row_count, np.nan) for variant in variants
    }
    ranks = {variant: np.full(row_count, np.nan) for variant in variants}
    blend = np.full(row_count, np.nan)
    fold_number = np.zeros(row_count, dtype=np.int8)
    importance_sums = {
        variant: np.zeros(len(features), dtype=float)
        for variant, (_, _, features) in variants.items()
    }
    fold_results: list[dict[str, object]] = []
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    total_started = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(x_exact, y), start=1
    ):
        fold_started = time.perf_counter()
        fold_seed = args.seed + fold
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        fold_number[valid_indices] = fold
        durations: dict[str, float] = {}
        for variant, (features, categorical, _) in variants.items():
            model = make_lightgbm(args.iterations, args.learning_rate, fold_seed)
            label = f"Fold {fold}/{args.folds} LightGBM control {variant}"
            started = time.perf_counter()
            with logger.step(f"{label} | Train"):
                model.fit(
                    features.iloc[train_indices],
                    y_train,
                    categorical_feature=categorical,
                    eval_X=features.iloc[valid_indices],
                    eval_y=y_valid,
                    eval_metric="auc",
                    callbacks=[
                        LightGBMProgressCallback(
                            logger, label, args.iterations, args.log_every
                        )
                    ],
                )
            current = model.predict_proba(features.iloc[valid_indices])[:, 1]
            probabilities[variant][valid_indices] = current
            ranks[variant][valid_indices] = percentile_rank(current)
            importance_sums[variant] += model.booster_.feature_importance(
                importance_type="gain"
            )
            durations[f"{variant}_seconds"] = time.perf_counter() - started

        current_blend = weighted_blend(
            ranks["exact"][valid_indices],
            ranks["base"][valid_indices],
            args.exact_weight,
        )
        blend[valid_indices] = current_blend
        fold_results.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "exact_auc": roc_auc_score(y_valid, ranks["exact"][valid_indices]),
                "base_auc": roc_auc_score(y_valid, ranks["base"][valid_indices]),
                "blend_auc": roc_auc_score(y_valid, current_blend),
                "durations": durations,
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
        elapsed = time.perf_counter() - total_started
        eta = elapsed / fold * (args.folds - fold)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | "
            f"blend={fold_results[-1]['blend_auc']:.10f} | "
            f"elapsed={format_duration(elapsed)}, ETA={format_duration(eta)}"
        )

    if not all(
        np.isfinite(values).all()
        for values in (*probabilities.values(), *ranks.values(), blend)
    ):
        raise ValueError("Control predictions are not finite")

    metrics = {
        "run_id": args.run_id,
        "comparison": "lightweight_lightgbm_family_control",
        "folds": args.folds,
        "seed": args.seed,
        "parameters": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "early_stopping": False,
            "exact_weight": args.exact_weight,
            "base_weight": 1 - args.exact_weight,
            "exact_features": exact_features,
            "base_features": base_features,
        },
        "components": {
            "exact_fold_rank_oof_auc": roc_auc_score(y, ranks["exact"]),
            "base_fold_rank_oof_auc": roc_auc_score(y, ranks["base"]),
            "component_spearman": float(
                spearmanr(ranks["exact"], ranks["base"]).statistic
            ),
        },
        "blend": {"candidate_fold_rank_oof_auc": roc_auc_score(y, blend)},
        "fold_results": fold_results,
        "elapsed_seconds": time.perf_counter() - total_started,
    }
    metrics_path = args.output_dir / f"metrics_{args.run_id}_lightweight_control.json"
    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    importance_path = args.output_dir / f"feature_importance_{args.run_id}.csv"
    with logger.step("Save metrics, OOF predictions and feature importance"):
        pd.DataFrame(
            {
                "row_index": np.arange(row_count),
                ID_COLUMN: train[ID_COLUMN],
                "target": y,
                "fold": fold_number,
                "candidate_exact_probability": probabilities["exact"],
                "candidate_base_probability": probabilities["base"],
                "candidate_exact_fold_rank": ranks["exact"],
                "candidate_base_fold_rank": ranks["base"],
                "candidate_blend_fold_rank": blend,
            }
        ).to_csv(oof_path, index=False)
        importance_rows = []
        for variant, (_, _, feature_names) in variants.items():
            for feature, gain in zip(
                feature_names, importance_sums[variant] / args.folds, strict=True
            ):
                importance_rows.append(
                    {"variant": variant, "feature": feature, "mean_gain": gain}
                )
        pd.DataFrame(importance_rows).to_csv(importance_path, index=False)
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        f"FINAL | exact={metrics['components']['exact_fold_rank_oof_auc']:.10f}, "
        f"base={metrics['components']['base_fold_rank_oof_auc']:.10f}, "
        f"blend={metrics['blend']['candidate_fold_rank_oof_auc']:.10f}"
    )
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"OOF predictions: {oof_path}")
    logger.finish()


if __name__ == "__main__":
    main()
