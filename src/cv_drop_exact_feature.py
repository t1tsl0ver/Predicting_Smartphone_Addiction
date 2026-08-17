from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from blend_cv import weighted_blend
from compare_oof import DEFAULT_TEST_SIZE
from cv_ensemble import prepare_catboost
from cv_family_blends import (
    bootstrap_summary,
    build_feature_variants,
)
from cv_head_to_head import make_catboost, make_lightgbm
from cv_lightgbm import ID_COLUMN, TARGET, percentile_rank, prepare_lightgbm
from progress_logging import (
    CatBoostProgressCallback,
    LightGBMProgressCallback,
    RunLogger,
    format_duration,
)
from trial_config import (
    TRIAL_CATBOOST_DEPTH,
    TRIAL_CATBOOST_L2_LEAF_REG,
    TRIAL_CATBOOST_LEARNING_RATE,
    TRIAL_EXACT_WEIGHT,
    TRIAL_FOLDS,
    TRIAL_ITERATIONS,
    TRIAL_LIGHTGBM_LEARNING_RATE,
    TRIAL_SEED,
)


DEFAULT_DROP_FEATURE = "gaming_hours_exact"
NON_INFERIORITY_TOLERANCE = 0.00010


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate one exact-value category and compare exact CatBoost and "
            "the LightGBM exact/base family blend against cached control OOF."
        )
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--control-oof",
        type=Path,
        default=Path("artifacts/oof_predictions_trial-023.csv"),
    )
    parser.add_argument(
        "--control-metrics",
        type=Path,
        default=Path("artifacts/metrics_trial-023_family_blends.json"),
    )
    parser.add_argument("--drop-exact-feature", default=DEFAULT_DROP_FEATURE)
    parser.add_argument("--folds", type=int, default=TRIAL_FOLDS)
    parser.add_argument("--seed", type=int, default=TRIAL_SEED)
    parser.add_argument("--iterations", type=int, default=TRIAL_ITERATIONS)
    parser.add_argument(
        "--lightgbm-learning-rate",
        type=float,
        default=TRIAL_LIGHTGBM_LEARNING_RATE,
    )
    parser.add_argument(
        "--catboost-learning-rate",
        type=float,
        default=TRIAL_CATBOOST_LEARNING_RATE,
    )
    parser.add_argument("--catboost-depth", type=int, default=TRIAL_CATBOOST_DEPTH)
    parser.add_argument(
        "--catboost-l2-leaf-reg",
        type=float,
        default=TRIAL_CATBOOST_L2_LEAF_REG,
    )
    parser.add_argument("--exact-weight", type=float, default=TRIAL_EXACT_WEIGHT)
    parser.add_argument(
        "--non-inferiority-tolerance",
        type=float,
        default=NON_INFERIORITY_TOLERANCE,
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-sample-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", type=str, default="trial-024")
    return parser.parse_args()


def remove_exact_feature(features: list[str], feature: str) -> list[str]:
    if not feature.endswith("_exact"):
        raise ValueError("The ablated feature must be an _exact categorical copy")
    if feature not in features:
        raise ValueError(f"Cannot ablate missing feature: {feature}")
    candidate = [column for column in features if column != feature]
    if len(candidate) != len(features) - 1:
        raise ValueError("Exactly one feature must be removed")
    return candidate


def expected_fold_numbers(
    target: pd.Series,
    folds: int,
    seed: int,
) -> np.ndarray:
    result = np.zeros(len(target), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(target), 1), dtype=np.int8)
    for fold, (_, valid_indices) in enumerate(splitter.split(dummy, target), start=1):
        result[valid_indices] = fold
    return result


def validate_control_artifacts(
    train: pd.DataFrame,
    control: pd.DataFrame,
    control_metrics: dict[str, object],
    folds: int,
    seed: int,
    iterations: int,
    exact_weight: float,
) -> None:
    required_columns = {
        "row_index",
        ID_COLUMN,
        "target",
        "fold",
        "catboost_exact_fold_rank",
        "lightgbm_exact_fold_rank",
        "lightgbm_base_fold_rank",
        "lightgbm_blend_fold_rank",
    }
    missing = sorted(required_columns - set(control.columns))
    if missing:
        raise ValueError(f"Control OOF is missing columns: {missing}")
    if len(control) != len(train):
        raise ValueError("Control OOF row count differs from training data")
    if not np.array_equal(control["row_index"], np.arange(len(train))):
        raise ValueError("Control OOF row order differs from training data")
    if not control[ID_COLUMN].equals(train[ID_COLUMN]):
        raise ValueError("Control OOF ids differ from training data")
    if not np.array_equal(control["target"].to_numpy(), train[TARGET].to_numpy()):
        raise ValueError("Control OOF targets differ from training data")

    expected_folds = expected_fold_numbers(train[TARGET], folds, seed)
    if not np.array_equal(control["fold"].to_numpy(dtype=np.int8), expected_folds):
        raise ValueError("Control OOF folds differ from the requested trial split")

    if int(control_metrics.get("folds", -1)) != folds:
        raise ValueError("Control metrics use a different fold count")
    if int(control_metrics.get("seed", -1)) != seed:
        raise ValueError("Control metrics use a different seed")
    shared = control_metrics.get("shared")
    blend = control_metrics.get("blend")
    if not isinstance(shared, dict) or int(shared.get("iterations", -1)) != iterations:
        raise ValueError("Control metrics use a different tree budget")
    if not isinstance(blend, dict) or not np.isclose(
        float(blend.get("exact_weight", np.nan)), exact_weight
    ):
        raise ValueError("Control metrics use a different blend weight")

    prediction_columns = [
        "catboost_exact_fold_rank",
        "lightgbm_exact_fold_rank",
        "lightgbm_base_fold_rank",
        "lightgbm_blend_fold_rank",
    ]
    if not np.isfinite(control[prediction_columns].to_numpy()).all():
        raise ValueError("Control OOF contains missing or infinite predictions")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if not 0 <= args.exact_weight <= 1:
        raise ValueError("--exact-weight must be between 0 and 1")
    if args.non_inferiority_tolerance < 0:
        raise ValueError("--non-inferiority-tolerance must be non-negative")
    if args.bootstrap_repeats < 2 or args.bootstrap_sample_size < 2:
        raise ValueError("Bootstrap repeats and sample size must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("cv-drop-exact-feature", args.output_dir / "logs", args.run_id)

    with logger.step("Load train and cached control artifacts"):
        train = pd.read_csv(args.train)
        control = pd.read_csv(args.control_oof)
        control_metrics = json.loads(args.control_metrics.read_text(encoding="utf-8"))
        if TARGET not in train or ID_COLUMN not in train:
            raise ValueError("Unexpected training schema")
        validate_control_artifacts(
            train,
            control,
            control_metrics,
            args.folds,
            args.seed,
            args.iterations,
            args.exact_weight,
        )

    with logger.step("Build candidate feature set"):
        exact_features, _ = build_feature_variants(train)
        candidate_features = remove_exact_feature(
            exact_features, args.drop_exact_feature
        )
        categorical = train[candidate_features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        empty = train[candidate_features].iloc[:0].copy()
        x_cat, _ = prepare_catboost(train[candidate_features], empty, categorical)
        x_lgb, _ = prepare_lightgbm(train[candidate_features], empty, categorical)
        y = train[TARGET]

    logger.info(
        f"Trial configuration | folds={args.folds}, seed={args.seed}, "
        f"trees={args.iterations}, early_stopping=False, "
        f"drop={args.drop_exact_feature}"
    )
    logger.info(
        f"Candidate features | total={len(candidate_features)}, "
        f"categorical={len(categorical)} | LightGBM blend="
        f"{args.exact_weight:.0%} candidate exact + "
        f"{1 - args.exact_weight:.0%} cached base"
    )

    row_count = len(train)
    cat_probability = np.full(row_count, np.nan)
    cat_rank = np.full(row_count, np.nan)
    lgb_probability = np.full(row_count, np.nan)
    lgb_rank = np.full(row_count, np.nan)
    lgb_blend = np.full(row_count, np.nan)
    cat_importance_sum = np.zeros(len(candidate_features))
    lgb_importance_sum = np.zeros(len(candidate_features))
    fold_results: list[dict[str, object]] = []

    control_fold = control["fold"].to_numpy(dtype=np.int8)
    control_cat = control["catboost_exact_fold_rank"].to_numpy()
    control_lgb_blend = control["lightgbm_blend_fold_rank"].to_numpy()
    control_lgb_base = control["lightgbm_base_fold_rank"].to_numpy()
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    total_started = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(x_lgb, y), start=1
    ):
        fold_started = time.perf_counter()
        fold_seed = args.seed + fold
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        if not np.all(control_fold[valid_indices] == fold):
            raise ValueError("Cached control fold assignment changed unexpectedly")

        logger.info(
            f"FOLD START | {fold}/{args.folds} | train={len(train_indices)}, "
            f"valid={len(valid_indices)}, seed={fold_seed}"
        )
        lgb_model = make_lightgbm(
            args.iterations, args.lightgbm_learning_rate, fold_seed
        )
        lgb_started = time.perf_counter()
        lgb_label = f"Fold {fold}/{args.folds} LightGBM candidate exact"
        with logger.step(f"{lgb_label} | Train"):
            lgb_model.fit(
                x_lgb.iloc[train_indices],
                y_train,
                categorical_feature=categorical,
                eval_X=x_lgb.iloc[valid_indices],
                eval_y=y_valid,
                eval_metric="auc",
                callbacks=[
                    LightGBMProgressCallback(
                        logger, lgb_label, args.iterations, args.log_every
                    )
                ],
            )
        current_lgb_probability = lgb_model.predict_proba(
            x_lgb.iloc[valid_indices]
        )[:, 1]
        current_lgb_rank = percentile_rank(current_lgb_probability)
        current_lgb_blend = weighted_blend(
            current_lgb_rank,
            control_lgb_base[valid_indices],
            args.exact_weight,
        )
        lgb_seconds = time.perf_counter() - lgb_started
        lgb_probability[valid_indices] = current_lgb_probability
        lgb_rank[valid_indices] = current_lgb_rank
        lgb_blend[valid_indices] = current_lgb_blend
        lgb_importance_sum += lgb_model.booster_.feature_importance(
            importance_type="gain"
        )

        cat_model = make_catboost(
            args.iterations,
            args.catboost_learning_rate,
            args.catboost_depth,
            args.catboost_l2_leaf_reg,
            fold_seed,
        )
        cat_started = time.perf_counter()
        cat_label = f"Fold {fold}/{args.folds} CatBoost candidate exact"
        with logger.step(f"{cat_label} | Train"):
            cat_model.fit(
                x_cat.iloc[train_indices],
                y_train,
                cat_features=categorical,
                eval_set=(x_cat.iloc[valid_indices], y_valid),
                use_best_model=False,
                callbacks=[
                    CatBoostProgressCallback(
                        logger, cat_label, args.iterations, args.log_every
                    )
                ],
            )
        current_cat_probability = cat_model.predict_proba(
            x_cat.iloc[valid_indices]
        )[:, 1]
        current_cat_rank = percentile_rank(current_cat_probability)
        cat_seconds = time.perf_counter() - cat_started
        cat_probability[valid_indices] = current_cat_probability
        cat_rank[valid_indices] = current_cat_rank
        cat_importance_sum += cat_model.get_feature_importance()

        cat_control_auc = roc_auc_score(y_valid, control_cat[valid_indices])
        cat_candidate_auc = roc_auc_score(y_valid, current_cat_rank)
        lgb_control_auc = roc_auc_score(
            y_valid, control_lgb_blend[valid_indices]
        )
        lgb_candidate_auc = roc_auc_score(y_valid, current_lgb_blend)
        fold_result: dict[str, object] = {
            "fold": fold,
            "train_rows": len(train_indices),
            "validation_rows": len(valid_indices),
            "catboost": {
                "control_auc": cat_control_auc,
                "candidate_auc": cat_candidate_auc,
                "candidate_minus_control": cat_candidate_auc - cat_control_auc,
            },
            "lightgbm_blend": {
                "control_auc": lgb_control_auc,
                "candidate_auc": lgb_candidate_auc,
                "candidate_minus_control": lgb_candidate_auc - lgb_control_auc,
            },
            "candidate_catboost_minus_candidate_lightgbm_blend": (
                cat_candidate_auc - lgb_candidate_auc
            ),
            "durations": {
                "lightgbm_candidate_seconds": lgb_seconds,
                "catboost_candidate_seconds": cat_seconds,
            },
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
        fold_results.append(fold_result)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | "
            f"CatBoost control={cat_control_auc:.10f}, "
            f"candidate={cat_candidate_auc:.10f}, "
            f"delta={cat_candidate_auc - cat_control_auc:+.10f} | "
            f"LGBM blend control={lgb_control_auc:.10f}, "
            f"candidate={lgb_candidate_auc:.10f}, "
            f"delta={lgb_candidate_auc - lgb_control_auc:+.10f}"
        )
        elapsed = time.perf_counter() - total_started
        eta = elapsed / fold * (args.folds - fold)
        logger.info(
            f"FOLD DONE | {fold}/{args.folds} | elapsed={format_duration(elapsed)} | "
            f"ETA={format_duration(eta)}"
        )

    for name, values in {
        "CatBoost candidate probability": cat_probability,
        "CatBoost candidate rank": cat_rank,
        "LightGBM candidate probability": lgb_probability,
        "LightGBM candidate rank": lgb_rank,
        "LightGBM candidate blend": lgb_blend,
    }.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing or infinite values")

    cat_control_auc = roc_auc_score(y, control_cat)
    cat_candidate_auc = roc_auc_score(y, cat_rank)
    cat_delta = cat_candidate_auc - cat_control_auc
    lgb_control_auc = roc_auc_score(y, control_lgb_blend)
    lgb_candidate_auc = roc_auc_score(y, lgb_blend)
    lgb_delta = lgb_candidate_auc - lgb_control_auc

    bootstrap_pairs = {
        "catboost_candidate_minus_control": (control_cat, cat_rank),
        "lightgbm_blend_candidate_minus_control": (
            control_lgb_blend,
            lgb_blend,
        ),
        "candidate_catboost_minus_candidate_lightgbm_blend": (
            lgb_blend,
            cat_rank,
        ),
    }
    bootstrap_results: dict[str, dict[str, float | int | str]] = {}
    target_array = y.to_numpy(dtype=np.int8)
    for offset, (name, (control_values, candidate_values)) in enumerate(
        bootstrap_pairs.items()
    ):
        with logger.step(f"Paired bootstrap | {name}"):
            bootstrap_results[name] = bootstrap_summary(
                target_array,
                control_values,
                candidate_values,
                args.bootstrap_repeats,
                args.bootstrap_sample_size,
                args.seed + offset,
            )

    control_fold_results = control_metrics.get("fold_results")
    if not isinstance(control_fold_results, list):
        raise ValueError("Control metrics do not contain fold timings")
    control_cat_seconds = sum(
        float(result["durations"]["catboost_exact_seconds"])
        for result in control_fold_results
    )
    control_lgb_seconds = sum(
        float(result["durations"]["lightgbm_exact_seconds"])
        for result in control_fold_results
    )
    candidate_cat_seconds = sum(
        float(result["durations"]["catboost_candidate_seconds"])
        for result in fold_results
    )
    candidate_lgb_seconds = sum(
        float(result["durations"]["lightgbm_candidate_seconds"])
        for result in fold_results
    )
    cat_fold_wins = all(
        float(result["catboost"]["candidate_minus_control"]) > 0
        for result in fold_results
    )
    lgb_fold_wins = all(
        float(result["lightgbm_blend"]["candidate_minus_control"]) > 0
        for result in fold_results
    )
    cat_bootstrap_positive = (
        float(
            bootstrap_results["catboost_candidate_minus_control"]["ci95_lower"]
        )
        > 0
    )
    lgb_bootstrap_positive = (
        float(
            bootstrap_results[
                "lightgbm_blend_candidate_minus_control"
            ]["ci95_lower"]
        )
        > 0
    )
    cross_family_interval = bootstrap_results[
        "candidate_catboost_minus_candidate_lightgbm_blend"
    ]
    cross_family_indistinguishable = (
        float(cross_family_interval["ci95_lower"]) <= 0
        <= float(cross_family_interval["ci95_upper"])
    )

    metrics = {
        "run_id": args.run_id,
        "comparison": "drop_one_exact_feature_ablation",
        "control_run_id": control_metrics.get("run_id"),
        "folds": args.folds,
        "seed": args.seed,
        "ablation": {
            "dropped_feature": args.drop_exact_feature,
            "control_feature_count": len(exact_features),
            "candidate_feature_count": len(candidate_features),
            "control_categorical_count": len(
                control_metrics["shared"]["exact_categorical_features"]
            ),
            "candidate_categorical_count": len(categorical),
        },
        "shared": {
            "iterations": args.iterations,
            "early_stopping": False,
            "candidate_features": candidate_features,
            "candidate_categorical_features": categorical,
            "lightgbm_blend_exact_weight": args.exact_weight,
            "lightgbm_blend_base_weight": 1 - args.exact_weight,
            "non_inferiority_tolerance": args.non_inferiority_tolerance,
        },
        "catboost": {
            "control_fold_rank_oof_auc": cat_control_auc,
            "candidate_fold_rank_oof_auc": cat_candidate_auc,
            "candidate_minus_control": cat_delta,
        },
        "lightgbm_blend": {
            "control_fold_rank_oof_auc": lgb_control_auc,
            "candidate_fold_rank_oof_auc": lgb_candidate_auc,
            "candidate_minus_control": lgb_delta,
        },
        "candidate_catboost_minus_candidate_lightgbm_blend": (
            cat_candidate_auc - lgb_candidate_auc
        ),
        "decision": {
            "catboost_point_estimate_improved": cat_delta > 0,
            "lightgbm_blend_point_estimate_improved": lgb_delta > 0,
            "catboost_non_inferior": (
                cat_delta >= -args.non_inferiority_tolerance
            ),
            "lightgbm_blend_non_inferior": (
                lgb_delta >= -args.non_inferiority_tolerance
            ),
            "catboost_won_all_folds": cat_fold_wins,
            "lightgbm_blend_won_all_folds": lgb_fold_wins,
            "catboost_row_bootstrap_interval_positive": cat_bootstrap_positive,
            "lightgbm_row_bootstrap_interval_positive": lgb_bootstrap_positive,
            "catboost_ablation_screening_accepted": (
                cat_delta > 0 and cat_fold_wins and cat_bootstrap_positive
            ),
            "lightgbm_ablation_screening_accepted": (
                lgb_delta > 0 and lgb_fold_wins and lgb_bootstrap_positive
            ),
            "shared_drop_screening_accepted": (
                cat_delta > 0
                and cat_fold_wins
                and cat_bootstrap_positive
                and lgb_delta > 0
                and lgb_fold_wins
                and lgb_bootstrap_positive
            ),
            "candidate_catboost_point_estimate_beats_lightgbm_blend": (
                cat_candidate_auc > lgb_candidate_auc
            ),
            "candidate_family_comparison_indistinguishable": (
                cross_family_indistinguishable
            ),
        },
        "timings": {
            "control_catboost_exact_seconds": control_cat_seconds,
            "candidate_catboost_seconds": candidate_cat_seconds,
            "catboost_speed_ratio_candidate_over_control": (
                candidate_cat_seconds / control_cat_seconds
            ),
            "control_lightgbm_exact_seconds": control_lgb_seconds,
            "candidate_lightgbm_seconds": candidate_lgb_seconds,
            "lightgbm_speed_ratio_candidate_over_control": (
                candidate_lgb_seconds / control_lgb_seconds
            ),
        },
        "paired_stratified_bootstrap": bootstrap_results,
        "fold_results": fold_results,
        "elapsed_seconds": time.perf_counter() - total_started,
    }

    metrics_path = args.output_dir / f"metrics_{args.run_id}_drop_exact.json"
    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    importance_path = args.output_dir / f"feature_importance_{args.run_id}.csv"
    with logger.step("Save metrics, OOF and candidate importance"):
        pd.DataFrame(
            {
                "row_index": np.arange(row_count),
                ID_COLUMN: train[ID_COLUMN],
                "target": y,
                "fold": control_fold,
                "catboost_control_fold_rank": control_cat,
                "catboost_candidate_probability": cat_probability,
                "catboost_candidate_fold_rank": cat_rank,
                "lightgbm_blend_control_fold_rank": control_lgb_blend,
                "lightgbm_candidate_exact_probability": lgb_probability,
                "lightgbm_candidate_exact_fold_rank": lgb_rank,
                "lightgbm_blend_candidate_fold_rank": lgb_blend,
            }
        ).to_csv(oof_path, index=False)
        pd.DataFrame(
            {
                "feature": candidate_features,
                "lightgbm_candidate_mean_gain": lgb_importance_sum / args.folds,
                "catboost_candidate_mean_importance": (
                    cat_importance_sum / args.folds
                ),
            }
        ).sort_values("catboost_candidate_mean_importance", ascending=False).to_csv(
            importance_path, index=False
        )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        f"FINAL | CatBoost control={cat_control_auc:.10f}, "
        f"candidate={cat_candidate_auc:.10f}, delta={cat_delta:+.10f}"
    )
    logger.info(
        f"FINAL | LightGBM blend control={lgb_control_auc:.10f}, "
        f"candidate={lgb_candidate_auc:.10f}, delta={lgb_delta:+.10f}"
    )
    logger.info(
        f"FINAL | Candidate CatBoost minus candidate LightGBM blend="
        f"{cat_candidate_auc - lgb_candidate_auc:+.10f}"
    )
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"OOF predictions: {oof_path}")
    logger.info(f"Feature importance: {importance_path}")
    logger.finish()


if __name__ == "__main__":
    main()
