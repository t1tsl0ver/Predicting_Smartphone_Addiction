from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from progress_logging import LightGBMProgressCallback, RunLogger, format_duration


TARGET = "addicted_label"
ID_COLUMN = "id"
OTHER_SCREEN_TIME = "other_screen_time_hours"
SCREEN_AWAKE_SHARE = "screen_awake_share"
WEEKEND_TO_DAILY_SCREEN_RATIO = "weekend_to_daily_screen_ratio"
SCREEN_SOCIAL_LOAD = "screen_social_load"
HIGH_STRESS_OTHER_SCREEN_PER_SLEEP = "high_stress_other_screen_per_sleep"
OVERALL_SCREEN_PRESSURE = "overall_screen_pressure"
CORE_SCREEN_PRESSURE = "core_screen_pressure"
EXACT_VALUE_SUFFIX = "_exact"
RAW_NUMERIC_FEATURES = (
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pure LightGBM OOF validation and optionally create a submission."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument(
        "--cat-smooth",
        type=float,
        default=10.0,
        help="LightGBM categorical smoothing strength.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=0,
        help="Stop a fold after this many rounds without improvement; 0 disables it.",
    )
    parser.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        help="Feature columns to exclude from both train and test.",
    )
    parser.add_argument(
        "--add-other-screen-time",
        action="store_true",
        help="Add daily minus social, gaming and work/study screen time.",
    )
    parser.add_argument(
        "--add-screen-awake-share",
        action="store_true",
        help="Add daily screen time divided by waking hours (24 minus sleep).",
    )
    parser.add_argument(
        "--add-weekend-to-daily-screen-ratio",
        action="store_true",
        help="Add weekend screen time divided by daily screen time.",
    )
    parser.add_argument(
        "--add-screen-social-load",
        action="store_true",
        help="Add the product of daily screen time and social-media hours.",
    )
    parser.add_argument(
        "--add-high-stress-other-screen-per-sleep",
        action="store_true",
        help="Add High-stress indicator times other screen time divided by sleep.",
    )
    parser.add_argument(
        "--add-overall-screen-pressure",
        action="store_true",
        help="Add mean normalized daily and weekend screen time.",
    )
    parser.add_argument(
        "--add-core-screen-pressure",
        action="store_true",
        help="Add mean normalized daily, weekend and social-media screen time.",
    )
    parser.add_argument(
        "--add-numeric-exact-categories",
        action="store_true",
        help=(
            "Add categorical copies of the exact values of retained raw numeric "
            "features while keeping their numeric versions."
        ),
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--submission",
        type=Path,
        default=None,
        help="Write a Kaggle submission only when an output path is explicitly given.",
    )
    parser.add_argument("--run-id", type=str, default="trial-008")
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def add_numeric_exact_categories(
    frame: pd.DataFrame,
    source_columns: list[str] | tuple[str, ...] = RAW_NUMERIC_FEATURES,
) -> list[str]:
    missing_columns = [column for column in source_columns if column not in frame]
    if missing_columns:
        raise ValueError(
            f"Cannot create exact-value categories; missing columns: {missing_columns}"
        )

    created: list[str] = []
    for column in source_columns:
        exact_column = f"{column}{EXACT_VALUE_SUFFIX}"
        frame[exact_column] = frame[column].astype("string")
        created.append(exact_column)
    return created


def prepare_lightgbm(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = train_features.copy()
    test_features = test_features.copy()
    for column in categorical:
        train_values = train_features[column].fillna("__MISSING__").astype(str)
        test_values = test_features[column].fillna("__MISSING__").astype(str)
        categories = sorted(set(train_values.unique()) | set(test_values.unique()))
        train_features[column] = pd.Categorical(train_values, categories=categories)
        test_features[column] = pd.Categorical(test_values, categories=categories)
    return train_features, test_features


def make_model(args: argparse.Namespace, fold_seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        cat_smooth=args.cat_smooth,
        random_state=fold_seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds must be non-negative")
    if args.cat_smooth < 0:
        raise ValueError("--cat-smooth must be non-negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("cv-lightgbm", args.output_dir / "logs", args.run_id)

    with logger.step("Load and validate input CSV files"):
        train = pd.read_csv(args.train)
        test = pd.read_csv(args.test)
        sample = pd.read_csv(args.sample_submission)
        if TARGET not in train or ID_COLUMN not in train or ID_COLUMN not in test:
            raise ValueError("Unexpected train/test schema")
        if list(sample.columns) != [ID_COLUMN, TARGET]:
            raise ValueError("Unexpected sample submission schema")
        if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
            raise ValueError("Test and sample submission ids differ")

    with logger.step("Prepare LightGBM features"):
        if args.add_other_screen_time:
            for frame in (train, test):
                frame[OTHER_SCREEN_TIME] = (
                    frame["daily_screen_time_hours"]
                    - frame["social_media_hours"]
                    - frame["gaming_hours"]
                    - frame["work_study_hours"]
                )
        if args.add_screen_awake_share:
            for frame in (train, test):
                awake_hours = 24 - frame["sleep_hours"]
                frame[SCREEN_AWAKE_SHARE] = (
                    frame["daily_screen_time_hours"] / awake_hours.where(awake_hours > 0)
                )
        if args.add_weekend_to_daily_screen_ratio:
            for frame in (train, test):
                valid_daily_screen = frame["daily_screen_time_hours"].where(
                    frame["daily_screen_time_hours"] > 0
                )
                frame[WEEKEND_TO_DAILY_SCREEN_RATIO] = (
                    frame["weekend_screen_time"] / valid_daily_screen
                )
        if args.add_screen_social_load:
            for frame in (train, test):
                frame[SCREEN_SOCIAL_LOAD] = (
                    frame["daily_screen_time_hours"]
                    * frame["social_media_hours"]
                )
        if args.add_high_stress_other_screen_per_sleep:
            if not args.add_other_screen_time:
                raise ValueError(
                    "--add-high-stress-other-screen-per-sleep requires "
                    "--add-other-screen-time"
                )
            stress_mapping = {"High": 1.0, "Medium": 0.0, "Low": 0.0}
            for frame in (train, test):
                high_stress = frame["stress_level"].map(stress_mapping)
                valid_sleep = frame["sleep_hours"].where(frame["sleep_hours"] > 0)
                frame[HIGH_STRESS_OTHER_SCREEN_PER_SLEEP] = (
                    high_stress * frame[OTHER_SCREEN_TIME] / valid_sleep
                )
        if args.add_overall_screen_pressure:
            for frame in (train, test):
                normalized_screen = pd.concat(
                    [
                        frame["daily_screen_time_hours"] / 15.0,
                        frame["weekend_screen_time"] / 18.0,
                    ],
                    axis=1,
                )
                frame[OVERALL_SCREEN_PRESSURE] = normalized_screen.mean(
                    axis=1, skipna=True
                )
        if args.add_core_screen_pressure:
            for frame in (train, test):
                normalized_screen = pd.concat(
                    [
                        frame["daily_screen_time_hours"] / 15.0,
                        frame["weekend_screen_time"] / 18.0,
                        frame["social_media_hours"] / 8.0,
                    ],
                    axis=1,
                )
                frame[CORE_SCREEN_PRESSURE] = normalized_screen.mean(
                    axis=1, skipna=True
                )
        exact_value_features: list[str] = []
        if args.add_numeric_exact_categories:
            exact_value_sources = [
                column
                for column in RAW_NUMERIC_FEATURES
                if column not in args.drop_features
            ]
            exact_value_features = add_numeric_exact_categories(
                train, exact_value_sources
            )
            test_exact_value_features = add_numeric_exact_categories(
                test, exact_value_sources
            )
            if exact_value_features != test_exact_value_features:
                raise ValueError("Train and test exact-value feature schemas differ")
        all_features = [
            column for column in train.columns if column not in (ID_COLUMN, TARGET)
        ]
        unknown_dropped = sorted(set(args.drop_features) - set(all_features))
        if unknown_dropped:
            raise ValueError(f"Unknown features requested for removal: {unknown_dropped}")
        features = [
            column for column in all_features if column not in args.drop_features
        ]
        test_features = [
            column
            for column in test.columns
            if column != ID_COLUMN and column not in args.drop_features
        ]
        if features != test_features:
            raise ValueError("Train and test feature schemas differ")
        y = train[TARGET]
        categorical = train[features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        x, x_test = prepare_lightgbm(
            train[features], test[features], categorical
        )

    logger.info(
        f"Data ready | train={len(train)}, test={len(test)}, features={len(features)}, "
        f"categorical={len(categorical)}, positive_rate={y.mean():.6f}"
    )
    logger.info(
        f"Configuration | folds={args.folds}, seed={args.seed}, "
        f"iterations={args.iterations}, learning_rate={args.learning_rate}, "
        f"num_leaves={args.num_leaves}, max_depth={args.max_depth}, "
        f"min_child_samples={args.min_child_samples}, "
        f"reg_alpha={args.reg_alpha}, reg_lambda={args.reg_lambda}, "
        f"cat_smooth={args.cat_smooth}, "
        f"early_stopping_rounds={args.early_stopping_rounds}, "
        f"drop_features={args.drop_features}, "
        f"add_other_screen_time={args.add_other_screen_time}, "
        f"add_screen_awake_share={args.add_screen_awake_share}, "
        "add_weekend_to_daily_screen_ratio="
        f"{args.add_weekend_to_daily_screen_ratio}, "
        f"add_screen_social_load={args.add_screen_social_load}, "
        "add_high_stress_other_screen_per_sleep="
        f"{args.add_high_stress_other_screen_per_sleep}, "
        f"add_overall_screen_pressure={args.add_overall_screen_pressure}, "
        f"add_core_screen_pressure={args.add_core_screen_pressure}, "
        "add_numeric_exact_categories="
        f"{args.add_numeric_exact_categories}"
    )
    if exact_value_features:
        logger.info(
            "Exact-value categories | "
            + ", ".join(
                f"{column}={train[column].nunique(dropna=True)}"
                for column in exact_value_features
            )
        )
    if args.add_other_screen_time:
        available_other = train[OTHER_SCREEN_TIME].dropna()
        logger.info(
            f"{OTHER_SCREEN_TIME} | available={len(available_other)}, "
            f"missing={train[OTHER_SCREEN_TIME].isna().mean():.2%}, "
            f"min={available_other.min():.4f}, "
            f"median={available_other.median():.4f}, "
            f"max={available_other.max():.4f}"
        )
    if args.add_screen_awake_share:
        available_share = train[SCREEN_AWAKE_SHARE].dropna()
        logger.info(
            f"{SCREEN_AWAKE_SHARE} | available={len(available_share)}, "
            f"missing={train[SCREEN_AWAKE_SHARE].isna().mean():.2%}, "
            f"min={available_share.min():.4f}, "
            f"median={available_share.median():.4f}, "
            f"max={available_share.max():.4f}"
        )
    if args.add_weekend_to_daily_screen_ratio:
        available_ratio = train[WEEKEND_TO_DAILY_SCREEN_RATIO].dropna()
        logger.info(
            f"{WEEKEND_TO_DAILY_SCREEN_RATIO} | available={len(available_ratio)}, "
            f"missing={train[WEEKEND_TO_DAILY_SCREEN_RATIO].isna().mean():.2%}, "
            f"min={available_ratio.min():.4f}, "
            f"median={available_ratio.median():.4f}, "
            f"max={available_ratio.max():.4f}"
        )
    if args.add_screen_social_load:
        available_load = train[SCREEN_SOCIAL_LOAD].dropna()
        logger.info(
            f"{SCREEN_SOCIAL_LOAD} | available={len(available_load)}, "
            f"missing={train[SCREEN_SOCIAL_LOAD].isna().mean():.2%}, "
            f"min={available_load.min():.4f}, "
            f"median={available_load.median():.4f}, "
            f"max={available_load.max():.4f}"
        )
    if args.add_high_stress_other_screen_per_sleep:
        available_stress_load = train[HIGH_STRESS_OTHER_SCREEN_PER_SLEEP].dropna()
        logger.info(
            f"{HIGH_STRESS_OTHER_SCREEN_PER_SLEEP} | "
            f"available={len(available_stress_load)}, "
            f"missing={train[HIGH_STRESS_OTHER_SCREEN_PER_SLEEP].isna().mean():.2%}, "
            f"min={available_stress_load.min():.4f}, "
            f"median={available_stress_load.median():.4f}, "
            f"max={available_stress_load.max():.4f}"
        )
    if args.add_overall_screen_pressure:
        available_pressure = train[OVERALL_SCREEN_PRESSURE].dropna()
        logger.info(
            f"{OVERALL_SCREEN_PRESSURE} | available={len(available_pressure)}, "
            f"missing={train[OVERALL_SCREEN_PRESSURE].isna().mean():.2%}, "
            f"min={available_pressure.min():.4f}, "
            f"median={available_pressure.median():.4f}, "
            f"max={available_pressure.max():.4f}"
        )
    if args.add_core_screen_pressure:
        available_pressure = train[CORE_SCREEN_PRESSURE].dropna()
        logger.info(
            f"{CORE_SCREEN_PRESSURE} | available={len(available_pressure)}, "
            f"missing={train[CORE_SCREEN_PRESSURE].isna().mean():.2%}, "
            f"min={available_pressure.min():.4f}, "
            f"median={available_pressure.median():.4f}, "
            f"max={available_pressure.max():.4f}"
        )

    row_count = len(train)
    test_count = len(test)
    raw_oof = np.full(row_count, np.nan)
    rank_oof = np.full(row_count, np.nan)
    fold_number = np.zeros(row_count, dtype=np.int8)
    test_probability_sum = np.zeros(test_count)
    test_rank_sum = np.zeros(test_count)
    importance_sum = np.zeros(len(features))
    fold_results: list[dict[str, float | int]] = []

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    total_start = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(x, y), start=1
    ):
        fold_start = time.perf_counter()
        fold_seed = args.seed + fold
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        logger.info(
            f"FOLD START | {fold}/{args.folds} | "
            f"train={len(train_indices)}, valid={len(valid_indices)}, seed={fold_seed}"
        )

        model = make_model(args, fold_seed)
        callbacks = [
            LightGBMProgressCallback(
                logger,
                f"Fold {fold}/{args.folds} LightGBM",
                args.iterations,
                args.log_every,
            )
        ]
        if args.early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(args.early_stopping_rounds, verbose=False)
            )
        with logger.step(f"Fold {fold}/{args.folds} | Train LightGBM"):
            model.fit(
                x.iloc[train_indices],
                y_train,
                categorical_feature=categorical,
                eval_X=x.iloc[valid_indices],
                eval_y=y_valid,
                eval_metric="auc",
                callbacks=callbacks,
            )

        best_iteration = model.best_iteration_ or model.n_iter_
        with logger.step(f"Fold {fold}/{args.folds} | Predictions and metrics"):
            valid_prediction = model.predict_proba(
                x.iloc[valid_indices], num_iteration=best_iteration
            )[:, 1]
            train_prediction = model.predict_proba(
                x.iloc[train_indices], num_iteration=best_iteration
            )[:, 1]
            test_prediction = model.predict_proba(
                x_test, num_iteration=best_iteration
            )[:, 1]
            valid_rank = percentile_rank(valid_prediction)
            test_rank = percentile_rank(test_prediction)
            train_auc = roc_auc_score(y_train, train_prediction)
            valid_auc = roc_auc_score(y_valid, valid_prediction)

        raw_oof[valid_indices] = valid_prediction
        rank_oof[valid_indices] = valid_rank
        fold_number[valid_indices] = fold
        test_probability_sum += test_prediction
        test_rank_sum += test_rank
        importance_sum += model.booster_.feature_importance(importance_type="gain")
        fold_results.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "iterations": best_iteration,
                "train_auc": train_auc,
                "validation_auc": valid_auc,
                "elapsed_seconds": time.perf_counter() - fold_start,
            }
        )
        elapsed = time.perf_counter() - total_start
        eta = elapsed / fold * (args.folds - fold)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | train_AUC={train_auc:.10f}, "
            f"validation_AUC={valid_auc:.10f}, best_iteration={best_iteration}"
        )
        logger.info(
            f"FOLD DONE | {fold}/{args.folds} ({100 * fold / args.folds:.1f}%) | "
            f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
        )

    if not np.isfinite(raw_oof).all() or not np.isfinite(rank_oof).all():
        raise ValueError("OOF predictions contain missing or infinite values")

    raw_oof_auc = roc_auc_score(y, raw_oof)
    rank_oof_auc = roc_auc_score(y, rank_oof)
    test_probability = test_probability_sum / args.folds
    test_rank = test_rank_sum / args.folds
    if rank_oof_auc > raw_oof_auc:
        selected_prediction = "fold_rank"
        submission_values = test_rank
    else:
        selected_prediction = "raw_probability"
        submission_values = test_probability

    elapsed_seconds = time.perf_counter() - total_start
    metrics = {
        "run_id": args.run_id,
        "model": "lightgbm",
        "folds": args.folds,
        "seed": args.seed,
        "parameters": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "cat_smooth": args.cat_smooth,
            "early_stopping_rounds": args.early_stopping_rounds,
            "drop_features": args.drop_features,
            "add_other_screen_time": args.add_other_screen_time,
            "add_screen_awake_share": args.add_screen_awake_share,
            "add_weekend_to_daily_screen_ratio": (
                args.add_weekend_to_daily_screen_ratio
            ),
            "add_screen_social_load": args.add_screen_social_load,
            "add_high_stress_other_screen_per_sleep": (
                args.add_high_stress_other_screen_per_sleep
            ),
            "add_overall_screen_pressure": args.add_overall_screen_pressure,
            "add_core_screen_pressure": args.add_core_screen_pressure,
            "add_numeric_exact_categories": args.add_numeric_exact_categories,
            "exact_value_features": exact_value_features,
        },
        "raw_oof_auc": raw_oof_auc,
        "fold_rank_oof_auc": rank_oof_auc,
        "selected_prediction": selected_prediction,
        "selection_rule": "higher OOF AUC between raw probabilities and fold ranks",
        "fold_results": fold_results,
        "elapsed_seconds": elapsed_seconds,
    }
    metrics_path = args.output_dir / f"metrics_{args.run_id}_cv.json"
    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    test_path = args.output_dir / f"test_predictions_{args.run_id}.csv"
    importance_path = args.output_dir / f"feature_importance_{args.run_id}.csv"

    digest: str | None = None
    with logger.step("Save metrics and predictions"):
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "row_index": train.index,
                "fold": fold_number,
                "target": y,
                "lightgbm_probability": raw_oof,
                "lightgbm_fold_rank": rank_oof,
            }
        ).to_csv(oof_path, index=False)
        pd.DataFrame(
            {
                ID_COLUMN: test[ID_COLUMN],
                "lightgbm_probability": test_probability,
                "lightgbm_cv_rank": test_rank,
            }
        ).to_csv(test_path, index=False)
        pd.DataFrame(
            {
                "feature": features,
                "mean_importance_gain": importance_sum / args.folds,
            }
        ).sort_values("mean_importance_gain", ascending=False).to_csv(
            importance_path, index=False
        )
        if args.submission is not None:
            submission = sample.copy()
            submission[TARGET] = submission_values
            submission.to_csv(args.submission, index=False)
            digest = hashlib.sha256(args.submission.read_bytes()).hexdigest()
            metadata = {
                "run_id": args.run_id,
                "selected_prediction": selected_prediction,
                "rows": len(submission),
                "minimum_prediction": float(submission_values.min()),
                "maximum_prediction": float(submission_values.max()),
                "sha256": digest,
            }
            metadata_path = args.submission.with_suffix(".metadata.json")
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    logger.info("OOF SUMMARY")
    logger.info(f"Raw probability OOF AUC={raw_oof_auc:.10f}")
    logger.info(f"Fold-rank OOF AUC={rank_oof_auc:.10f}")
    logger.info(f"Selected prediction={selected_prediction}")
    if args.submission is None:
        logger.info("Submission not requested; Kaggle CSV was not created")
    else:
        logger.info(f"Submission saved to {args.submission}")
        logger.info(f"SHA-256={digest}")
    logger.info(f"Total CV time={format_duration(elapsed_seconds)}")
    logger.finish()


if __name__ == "__main__":
    main()
