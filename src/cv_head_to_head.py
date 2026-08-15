from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from compare_oof import DEFAULT_TEST_SIZE, paired_stratified_bootstrap_deltas
from cv_ensemble import prepare_catboost
from cv_lightgbm import (
    HIGH_STRESS_OTHER_SCREEN_PER_SLEEP,
    ID_COLUMN,
    OTHER_SCREEN_TIME,
    RAW_NUMERIC_FEATURES,
    SCREEN_SOCIAL_LOAD,
    TARGET,
    add_numeric_exact_categories,
    percentile_rank,
    prepare_lightgbm,
)
from progress_logging import (
    CatBoostProgressCallback,
    LightGBMProgressCallback,
    RunLogger,
    format_duration,
)


DROP_FEATURES = ("gender", "stress_level")
DEFAULT_TOLERANCE = 0.00010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare standalone CatBoost and LightGBM on identical folds, "
            "features and fixed tree budgets."
        )
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
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.1)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.08)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--success-tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument("--bootstrap-sample-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", type=str, default="trial-022")
    return parser.parse_args()


def catboost_caught_up(
    catboost_auc: float,
    lightgbm_auc: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative")
    return lightgbm_auc - catboost_auc <= tolerance + 1e-12


def add_accepted_features(frame: pd.DataFrame) -> list[str]:
    required = {
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "stress_level",
        *RAW_NUMERIC_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot build accepted features; missing columns: {missing}")

    frame[OTHER_SCREEN_TIME] = (
        frame["daily_screen_time_hours"]
        - frame["social_media_hours"]
        - frame["gaming_hours"]
        - frame["work_study_hours"]
    )
    frame[SCREEN_SOCIAL_LOAD] = (
        frame["daily_screen_time_hours"] * frame["social_media_hours"]
    )
    high_stress = frame["stress_level"].map(
        {"High": 1.0, "Medium": 0.0, "Low": 0.0}
    )
    valid_sleep = frame["sleep_hours"].where(frame["sleep_hours"] > 0)
    frame[HIGH_STRESS_OTHER_SCREEN_PER_SLEEP] = (
        high_stress * frame[OTHER_SCREEN_TIME] / valid_sleep
    )
    return add_numeric_exact_categories(frame)


def make_lightgbm(iterations: int, learning_rate: float, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=learning_rate,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        cat_smooth=10.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def make_catboost(
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    seed: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        thread_count=-1,
        verbose=False,
        allow_writing_files=False,
    )


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if args.bootstrap_repeats < 2 or args.bootstrap_sample_size < 2:
        raise ValueError("Bootstrap repeats and sample size must be at least 2")
    if args.success_tolerance < 0:
        raise ValueError("--success-tolerance must be non-negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("cv-head-to-head", args.output_dir / "logs", args.run_id)

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

    with logger.step("Build identical accepted features for both models"):
        train_exact = add_accepted_features(train)
        test_exact = add_accepted_features(test)
        if train_exact != test_exact:
            raise ValueError("Train and test exact-value feature schemas differ")

        all_features = [
            column for column in train.columns if column not in (ID_COLUMN, TARGET)
        ]
        features = [column for column in all_features if column not in DROP_FEATURES]
        test_features = [
            column
            for column in test.columns
            if column != ID_COLUMN and column not in DROP_FEATURES
        ]
        if features != test_features:
            raise ValueError("Train and test feature schemas differ")

        y = train[TARGET]
        categorical = train[features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        x_cat, test_cat = prepare_catboost(
            train[features], test[features], categorical
        )
        x_lgb, test_lgb = prepare_lightgbm(
            train[features], test[features], categorical
        )

    logger.info(
        f"Data ready | train={len(train)}, test={len(test)}, features={len(features)}, "
        f"categorical={len(categorical)}, positive_rate={y.mean():.6f}"
    )
    logger.info(
        f"Shared CV | folds={args.folds}, shuffle=True, seed={args.seed}, "
        f"fixed_iterations={args.iterations}, early_stopping=False"
    )
    logger.info(
        "LightGBM | learning_rate="
        f"{args.lightgbm_learning_rate}, leaves=31, min_child_samples=20, cat_smooth=10"
    )
    logger.info(
        f"CatBoost | learning_rate={args.catboost_learning_rate}, "
        f"depth={args.catboost_depth}, l2_leaf_reg={args.catboost_l2_leaf_reg}"
    )
    logger.info(
        f"Success criterion | CatBoost raw OOF no more than "
        f"{args.success_tolerance:.5f} below LightGBM raw OOF"
    )

    row_count = len(train)
    test_count = len(test)
    lgb_oof = np.full(row_count, np.nan)
    cat_oof = np.full(row_count, np.nan)
    lgb_rank_oof = np.full(row_count, np.nan)
    cat_rank_oof = np.full(row_count, np.nan)
    fold_number = np.zeros(row_count, dtype=np.int8)
    lgb_test_probability_sum = np.zeros(test_count)
    cat_test_probability_sum = np.zeros(test_count)
    lgb_test_rank_sum = np.zeros(test_count)
    cat_test_rank_sum = np.zeros(test_count)
    lgb_importance_sum = np.zeros(len(features))
    cat_importance_sum = np.zeros(len(features))
    fold_results: list[dict[str, float | int]] = []

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    total_start = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(x_lgb, y), start=1
    ):
        fold_started = time.perf_counter()
        fold_seed = args.seed + fold
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        logger.info(
            f"FOLD START | {fold}/{args.folds} | train={len(train_indices)}, "
            f"valid={len(valid_indices)}, seed={fold_seed}"
        )

        lgb_model = make_lightgbm(
            args.iterations, args.lightgbm_learning_rate, fold_seed
        )
        with logger.step(f"Fold {fold}/{args.folds} | Train LightGBM"):
            lgb_model.fit(
                x_lgb.iloc[train_indices],
                y_train,
                categorical_feature=categorical,
                eval_X=x_lgb.iloc[valid_indices],
                eval_y=y_valid,
                eval_metric="auc",
                callbacks=[
                    LightGBMProgressCallback(
                        logger,
                        f"Fold {fold}/{args.folds} LightGBM",
                        args.iterations,
                        args.log_every,
                    )
                ],
            )
        with logger.step(f"Fold {fold}/{args.folds} | LightGBM predictions"):
            lgb_valid = lgb_model.predict_proba(x_lgb.iloc[valid_indices])[:, 1]
            lgb_train = lgb_model.predict_proba(x_lgb.iloc[train_indices])[:, 1]
            lgb_test = lgb_model.predict_proba(test_lgb)[:, 1]

        cat_model = make_catboost(
            args.iterations,
            args.catboost_learning_rate,
            args.catboost_depth,
            args.catboost_l2_leaf_reg,
            fold_seed,
        )
        with logger.step(f"Fold {fold}/{args.folds} | Train CatBoost"):
            cat_model.fit(
                x_cat.iloc[train_indices],
                y_train,
                cat_features=categorical,
                eval_set=(x_cat.iloc[valid_indices], y_valid),
                use_best_model=False,
                callbacks=[
                    CatBoostProgressCallback(
                        logger,
                        f"Fold {fold}/{args.folds} CatBoost",
                        args.iterations,
                        args.log_every,
                    )
                ],
            )
        with logger.step(f"Fold {fold}/{args.folds} | CatBoost predictions"):
            cat_valid = cat_model.predict_proba(x_cat.iloc[valid_indices])[:, 1]
            cat_train = cat_model.predict_proba(x_cat.iloc[train_indices])[:, 1]
            cat_test = cat_model.predict_proba(test_cat)[:, 1]

        lgb_valid_auc = roc_auc_score(y_valid, lgb_valid)
        cat_valid_auc = roc_auc_score(y_valid, cat_valid)
        lgb_train_auc = roc_auc_score(y_train, lgb_train)
        cat_train_auc = roc_auc_score(y_train, cat_train)
        lgb_valid_rank = percentile_rank(lgb_valid)
        cat_valid_rank = percentile_rank(cat_valid)

        lgb_oof[valid_indices] = lgb_valid
        cat_oof[valid_indices] = cat_valid
        lgb_rank_oof[valid_indices] = lgb_valid_rank
        cat_rank_oof[valid_indices] = cat_valid_rank
        fold_number[valid_indices] = fold
        lgb_test_probability_sum += lgb_test
        cat_test_probability_sum += cat_test
        lgb_test_rank_sum += percentile_rank(lgb_test)
        cat_test_rank_sum += percentile_rank(cat_test)
        lgb_importance_sum += lgb_model.booster_.feature_importance(
            importance_type="gain"
        )
        cat_importance_sum += cat_model.get_feature_importance()

        fold_result = {
            "fold": fold,
            "train_rows": len(train_indices),
            "validation_rows": len(valid_indices),
            "lightgbm_train_auc": lgb_train_auc,
            "catboost_train_auc": cat_train_auc,
            "lightgbm_validation_auc": lgb_valid_auc,
            "catboost_validation_auc": cat_valid_auc,
            "catboost_minus_lightgbm": cat_valid_auc - lgb_valid_auc,
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
        fold_results.append(fold_result)
        elapsed = time.perf_counter() - total_start
        eta = elapsed / fold * (args.folds - fold)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | LGBM={lgb_valid_auc:.10f}, "
            f"CatBoost={cat_valid_auc:.10f}, "
            f"CatBoost-LGBM={cat_valid_auc - lgb_valid_auc:+.10f}"
        )
        logger.info(
            f"FOLD DONE | {fold}/{args.folds} ({100 * fold / args.folds:.1f}%) | "
            f"elapsed={format_duration(elapsed)} | ETA={format_duration(eta)}"
        )

    for name, values in {
        "LightGBM raw OOF": lgb_oof,
        "CatBoost raw OOF": cat_oof,
        "LightGBM rank OOF": lgb_rank_oof,
        "CatBoost rank OOF": cat_rank_oof,
    }.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing or infinite values")

    lgb_raw_auc = roc_auc_score(y, lgb_oof)
    cat_raw_auc = roc_auc_score(y, cat_oof)
    lgb_rank_auc = roc_auc_score(y, lgb_rank_oof)
    cat_rank_auc = roc_auc_score(y, cat_rank_oof)
    raw_gap = cat_raw_auc - lgb_raw_auc
    rank_gap = cat_rank_auc - lgb_rank_auc
    caught_up = catboost_caught_up(
        cat_raw_auc, lgb_raw_auc, args.success_tolerance
    )

    logger.info("START | Paired stratified bootstrap CatBoost minus LightGBM")
    bootstrap_started = time.perf_counter()
    bootstrap_deltas = paired_stratified_bootstrap_deltas(
        y.to_numpy(dtype=np.int8),
        lgb_oof,
        cat_oof,
        args.bootstrap_repeats,
        args.bootstrap_sample_size,
        args.seed,
    )
    logger.info(
        "DONE  | Paired stratified bootstrap CatBoost minus LightGBM | "
        f"duration={format_duration(time.perf_counter() - bootstrap_started)}"
    )

    elapsed_seconds = time.perf_counter() - total_start
    metrics = {
        "run_id": args.run_id,
        "comparison": "standalone_catboost_vs_standalone_lightgbm",
        "folds": args.folds,
        "seed": args.seed,
        "shared": {
            "iterations": args.iterations,
            "early_stopping": False,
            "features": features,
            "categorical_features": categorical,
            "drop_features": list(DROP_FEATURES),
        },
        "lightgbm_parameters": {
            "learning_rate": args.lightgbm_learning_rate,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "cat_smooth": 10.0,
        },
        "catboost_parameters": {
            "learning_rate": args.catboost_learning_rate,
            "depth": args.catboost_depth,
            "l2_leaf_reg": args.catboost_l2_leaf_reg,
            "loss_function": "Logloss",
        },
        "lightgbm_raw_oof_auc": lgb_raw_auc,
        "catboost_raw_oof_auc": cat_raw_auc,
        "catboost_minus_lightgbm_raw_oof_auc": raw_gap,
        "lightgbm_fold_rank_oof_auc": lgb_rank_auc,
        "catboost_fold_rank_oof_auc": cat_rank_auc,
        "catboost_minus_lightgbm_fold_rank_oof_auc": rank_gap,
        "prediction_pearson_correlation": float(np.corrcoef(lgb_oof, cat_oof)[0, 1]),
        "prediction_spearman_correlation": float(
            spearmanr(lgb_oof, cat_oof).statistic
        ),
        "success_criterion": {
            "metric": "raw_oof_auc",
            "maximum_allowed_catboost_deficit": args.success_tolerance,
            "catboost_caught_up": caught_up,
        },
        "paired_stratified_bootstrap": {
            "scope": "row-sampling uncertainty only; not training or LB-split uncertainty",
            "candidate_minus_control": "catboost_minus_lightgbm",
            "repeats": args.bootstrap_repeats,
            "sample_size": args.bootstrap_sample_size,
            "seed": args.seed,
            "mean_auc_delta": float(bootstrap_deltas.mean()),
            "standard_deviation": float(bootstrap_deltas.std(ddof=1)),
            "ci95_lower": float(np.quantile(bootstrap_deltas, 0.025)),
            "ci95_upper": float(np.quantile(bootstrap_deltas, 0.975)),
            "probability_delta_positive": float(np.mean(bootstrap_deltas > 0)),
        },
        "fold_results": fold_results,
        "elapsed_seconds": elapsed_seconds,
    }

    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    test_path = args.output_dir / f"test_predictions_{args.run_id}.csv"
    metrics_path = args.output_dir / f"metrics_{args.run_id}_head_to_head.json"
    importance_path = args.output_dir / f"feature_importance_{args.run_id}.csv"

    with logger.step("Save metrics and prediction artifacts"):
        pd.DataFrame(
            {
                "row_index": np.arange(row_count),
                ID_COLUMN: train[ID_COLUMN],
                "target": y,
                "fold": fold_number,
                "lightgbm_probability": lgb_oof,
                "catboost_probability": cat_oof,
                "lightgbm_fold_rank": lgb_rank_oof,
                "catboost_fold_rank": cat_rank_oof,
            }
        ).to_csv(oof_path, index=False)
        pd.DataFrame(
            {
                ID_COLUMN: test[ID_COLUMN],
                "lightgbm_probability": lgb_test_probability_sum / args.folds,
                "catboost_probability": cat_test_probability_sum / args.folds,
                "lightgbm_fold_rank": lgb_test_rank_sum / args.folds,
                "catboost_fold_rank": cat_test_rank_sum / args.folds,
            }
        ).to_csv(test_path, index=False)
        pd.DataFrame(
            {
                "feature": features,
                "lightgbm_mean_gain": lgb_importance_sum / args.folds,
                "catboost_mean_importance": cat_importance_sum / args.folds,
            }
        ).sort_values("lightgbm_mean_gain", ascending=False).to_csv(
            importance_path, index=False
        )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        f"FINAL | LGBM raw={lgb_raw_auc:.10f}, CatBoost raw={cat_raw_auc:.10f}, "
        f"gap={raw_gap:+.10f}"
    )
    logger.info(
        f"FINAL | LGBM fold-rank={lgb_rank_auc:.10f}, "
        f"CatBoost fold-rank={cat_rank_auc:.10f}, gap={rank_gap:+.10f}"
    )
    logger.info(
        f"DECISION | CatBoost caught up within {args.success_tolerance:.5f}: "
        f"{caught_up}"
    )
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"OOF predictions: {oof_path}")
    logger.info(f"Test predictions (no submission): {test_path}")
    logger.info(f"Feature importance: {importance_path}")
    logger.finish()


if __name__ == "__main__":
    main()
