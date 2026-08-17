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
from compare_oof import DEFAULT_TEST_SIZE, paired_stratified_bootstrap_deltas
from cv_ensemble import prepare_catboost
from cv_head_to_head import (
    DROP_FEATURES,
    add_accepted_features,
    make_catboost,
    make_lightgbm,
)
from cv_lightgbm import ID_COLUMN, TARGET, percentile_rank, prepare_lightgbm
from progress_logging import (
    CatBoostProgressCallback,
    LightGBMProgressCallback,
    RunLogger,
    format_duration,
)


DEFAULT_EXACT_WEIGHT = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CatBoost and LightGBM rank blends of exact-value and "
            "base feature variants on shared CV folds."
        )
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.10)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.08)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--exact-weight", type=float, default=DEFAULT_EXACT_WEIGHT)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-sample-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", type=str, default="trial-023")
    return parser.parse_args()


def build_feature_variants(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Add accepted features and return exact and non-exact feature lists."""
    exact_columns = set(add_accepted_features(frame))
    exact_features = [
        column
        for column in frame.columns
        if column not in (ID_COLUMN, TARGET, *DROP_FEATURES)
    ]
    base_features = [
        column for column in exact_features if column not in exact_columns
    ]
    if not exact_columns:
        raise ValueError("The exact feature variant must add categorical copies")
    if len(exact_features) <= len(base_features):
        raise ValueError("The exact feature variant must contain additional features")
    return exact_features, base_features


def fold_rank_blend(
    exact_probabilities: np.ndarray,
    base_probabilities: np.ndarray,
    exact_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exact_ranks = percentile_rank(exact_probabilities)
    base_ranks = percentile_rank(base_probabilities)
    blend = weighted_blend(exact_ranks, base_ranks, exact_weight)
    return exact_ranks, base_ranks, blend


def bootstrap_summary(
    target: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    repeats: int,
    sample_size: int,
    seed: int,
) -> dict[str, float | int | str]:
    deltas = paired_stratified_bootstrap_deltas(
        target,
        control,
        candidate,
        repeats,
        sample_size,
        seed,
    )
    return {
        "scope": "row-sampling uncertainty only; not training or fold uncertainty",
        "repeats": repeats,
        "sample_size": sample_size,
        "seed": seed,
        "mean_auc_delta": float(deltas.mean()),
        "standard_deviation": float(deltas.std(ddof=1)),
        "ci95_lower": float(np.quantile(deltas, 0.025)),
        "ci95_upper": float(np.quantile(deltas, 0.975)),
        "probability_delta_positive": float(np.mean(deltas > 0)),
    }


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if not 0 <= args.exact_weight <= 1:
        raise ValueError("--exact-weight must be between 0 and 1")
    if args.bootstrap_repeats < 2 or args.bootstrap_sample_size < 2:
        raise ValueError("Bootstrap repeats and sample size must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("cv-family-blends", args.output_dir / "logs", args.run_id)

    with logger.step("Load and validate training CSV"):
        train = pd.read_csv(args.train)
        if TARGET not in train or ID_COLUMN not in train:
            raise ValueError("Unexpected training schema")
        if not set(train[TARGET].unique()).issubset({0, 1}):
            raise ValueError(f"{TARGET!r} must be binary")

    with logger.step("Build exact and base feature variants"):
        exact_features, base_features = build_feature_variants(train)
        y = train[TARGET]
        exact_categorical = train[exact_features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        base_categorical = train[base_features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        empty_exact = train[exact_features].iloc[:0].copy()
        x_cat_exact, _ = prepare_catboost(
            train[exact_features], empty_exact, exact_categorical
        )
        x_lgb_exact, _ = prepare_lightgbm(
            train[exact_features], empty_exact, exact_categorical
        )
        x_cat_base = x_cat_exact[base_features]
        x_lgb_base = x_lgb_exact[base_features]

    logger.info(
        f"Data ready | rows={len(train)}, folds={args.folds}, seed={args.seed}, "
        f"exact_features={len(exact_features)}, base_features={len(base_features)}, "
        f"exact_categorical={len(exact_categorical)}, "
        f"base_categorical={len(base_categorical)}"
    )
    logger.info(
        f"Fixed blend | exact_weight={args.exact_weight:.2f}, "
        f"base_weight={1 - args.exact_weight:.2f}, fold percentile ranks"
    )
    logger.info(
        f"Trial configuration | trees={args.iterations}, early_stopping=False | "
        f"LightGBM LR={args.lightgbm_learning_rate}, leaves=31 | "
        f"CatBoost LR={args.catboost_learning_rate}, depth={args.catboost_depth}, "
        f"L2={args.catboost_l2_leaf_reg}"
    )

    row_count = len(train)
    probabilities = {
        family: {
            variant: np.full(row_count, np.nan)
            for variant in ("exact", "base")
        }
        for family in ("lightgbm", "catboost")
    }
    fold_ranks = {
        family: {
            variant: np.full(row_count, np.nan)
            for variant in ("exact", "base")
        }
        for family in ("lightgbm", "catboost")
    }
    blend_oof = {
        family: np.full(row_count, np.nan)
        for family in ("lightgbm", "catboost")
    }
    fold_number = np.zeros(row_count, dtype=np.int8)
    fold_results: list[dict[str, object]] = []

    variants = {
        "exact": {
            "catboost": x_cat_exact,
            "lightgbm": x_lgb_exact,
            "categorical": exact_categorical,
        },
        "base": {
            "catboost": x_cat_base,
            "lightgbm": x_lgb_base,
            "categorical": base_categorical,
        },
    }
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    total_started = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(x_lgb_exact, y), start=1
    ):
        fold_started = time.perf_counter()
        fold_seed = args.seed + fold
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        fold_number[valid_indices] = fold
        durations: dict[str, float] = {}
        logger.info(
            f"FOLD START | {fold}/{args.folds} | train={len(train_indices)}, "
            f"valid={len(valid_indices)}, seed={fold_seed}"
        )

        for variant_name, variant in variants.items():
            x_lgb = variant["lightgbm"]
            categorical = variant["categorical"]
            lgb_model = make_lightgbm(
                args.iterations, args.lightgbm_learning_rate, fold_seed
            )
            label = f"Fold {fold}/{args.folds} LightGBM {variant_name}"
            started = time.perf_counter()
            with logger.step(f"{label} | Train"):
                lgb_model.fit(
                    x_lgb.iloc[train_indices],
                    y_train,
                    categorical_feature=categorical,
                    eval_X=x_lgb.iloc[valid_indices],
                    eval_y=y_valid,
                    eval_metric="auc",
                    callbacks=[
                        LightGBMProgressCallback(
                            logger, label, args.iterations, args.log_every
                        )
                    ],
                )
            lgb_valid = lgb_model.predict_proba(x_lgb.iloc[valid_indices])[:, 1]
            durations[f"lightgbm_{variant_name}_seconds"] = (
                time.perf_counter() - started
            )
            probabilities["lightgbm"][variant_name][valid_indices] = lgb_valid

        for variant_name, variant in variants.items():
            x_cat = variant["catboost"]
            categorical = variant["categorical"]
            cat_model = make_catboost(
                args.iterations,
                args.catboost_learning_rate,
                args.catboost_depth,
                args.catboost_l2_leaf_reg,
                fold_seed,
            )
            label = f"Fold {fold}/{args.folds} CatBoost {variant_name}"
            started = time.perf_counter()
            with logger.step(f"{label} | Train"):
                cat_model.fit(
                    x_cat.iloc[train_indices],
                    y_train,
                    cat_features=categorical,
                    eval_set=(x_cat.iloc[valid_indices], y_valid),
                    use_best_model=False,
                    callbacks=[
                        CatBoostProgressCallback(
                            logger, label, args.iterations, args.log_every
                        )
                    ],
                )
            cat_valid = cat_model.predict_proba(x_cat.iloc[valid_indices])[:, 1]
            durations[f"catboost_{variant_name}_seconds"] = (
                time.perf_counter() - started
            )
            probabilities["catboost"][variant_name][valid_indices] = cat_valid

        family_results: dict[str, dict[str, float]] = {}
        for family in ("lightgbm", "catboost"):
            exact_valid = probabilities[family]["exact"][valid_indices]
            base_valid = probabilities[family]["base"][valid_indices]
            exact_rank, base_rank, blend = fold_rank_blend(
                exact_valid, base_valid, args.exact_weight
            )
            fold_ranks[family]["exact"][valid_indices] = exact_rank
            fold_ranks[family]["base"][valid_indices] = base_rank
            blend_oof[family][valid_indices] = blend
            family_results[family] = {
                "exact_auc": roc_auc_score(y_valid, exact_valid),
                "base_auc": roc_auc_score(y_valid, base_valid),
                "blend_auc": roc_auc_score(y_valid, blend),
            }
            family_results[family]["blend_minus_exact"] = (
                family_results[family]["blend_auc"]
                - family_results[family]["exact_auc"]
            )

        fold_result: dict[str, object] = {
            "fold": fold,
            "train_rows": len(train_indices),
            "validation_rows": len(valid_indices),
            "lightgbm": family_results["lightgbm"],
            "catboost": family_results["catboost"],
            "catboost_blend_minus_lightgbm_blend": (
                family_results["catboost"]["blend_auc"]
                - family_results["lightgbm"]["blend_auc"]
            ),
            "durations": durations,
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
        fold_results.append(fold_result)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | "
            f"LGBM exact={family_results['lightgbm']['exact_auc']:.10f}, "
            f"blend={family_results['lightgbm']['blend_auc']:.10f}, "
            f"delta={family_results['lightgbm']['blend_minus_exact']:+.10f} | "
            f"CatBoost exact={family_results['catboost']['exact_auc']:.10f}, "
            f"blend={family_results['catboost']['blend_auc']:.10f}, "
            f"delta={family_results['catboost']['blend_minus_exact']:+.10f}"
        )
        elapsed = time.perf_counter() - total_started
        eta = elapsed / fold * (args.folds - fold)
        logger.info(
            f"FOLD DONE | {fold}/{args.folds} | elapsed={format_duration(elapsed)} | "
            f"ETA={format_duration(eta)}"
        )

    all_predictions = {
        **{
            f"{family}_{variant}_probability": probabilities[family][variant]
            for family in probabilities
            for variant in probabilities[family]
        },
        **{
            f"{family}_{variant}_fold_rank": fold_ranks[family][variant]
            for family in fold_ranks
            for variant in fold_ranks[family]
        },
        **{f"{family}_blend_fold_rank": blend_oof[family] for family in blend_oof},
    }
    for name, values in all_predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing or infinite values")

    family_metrics: dict[str, dict[str, float]] = {}
    for family in ("lightgbm", "catboost"):
        exact_probability = probabilities[family]["exact"]
        base_probability = probabilities[family]["base"]
        exact_rank = fold_ranks[family]["exact"]
        base_rank = fold_ranks[family]["base"]
        blend = blend_oof[family]
        exact_rank_auc = roc_auc_score(y, exact_rank)
        base_rank_auc = roc_auc_score(y, base_rank)
        blend_auc = roc_auc_score(y, blend)
        family_metrics[family] = {
            "exact_raw_oof_auc": roc_auc_score(y, exact_probability),
            "base_raw_oof_auc": roc_auc_score(y, base_probability),
            "exact_fold_rank_oof_auc": exact_rank_auc,
            "base_fold_rank_oof_auc": base_rank_auc,
            "blend_fold_rank_oof_auc": blend_auc,
            "blend_minus_exact": blend_auc - exact_rank_auc,
            "blend_minus_best_component": blend_auc
            - max(exact_rank_auc, base_rank_auc),
            "component_pearson_correlation": float(
                np.corrcoef(exact_rank, base_rank)[0, 1]
            ),
            "component_spearman_correlation": float(
                spearmanr(exact_rank, base_rank).statistic
            ),
        }

    target_array = y.to_numpy(dtype=np.int8)
    bootstrap_comparisons = {
        "catboost_blend_minus_catboost_exact": (
            fold_ranks["catboost"]["exact"],
            blend_oof["catboost"],
        ),
        "lightgbm_blend_minus_lightgbm_exact": (
            fold_ranks["lightgbm"]["exact"],
            blend_oof["lightgbm"],
        ),
        "catboost_blend_minus_lightgbm_blend": (
            blend_oof["lightgbm"],
            blend_oof["catboost"],
        ),
    }
    bootstrap_results: dict[str, dict[str, float | int | str]] = {}
    for offset, (name, (control, candidate)) in enumerate(
        bootstrap_comparisons.items()
    ):
        with logger.step(f"Paired bootstrap | {name}"):
            bootstrap_results[name] = bootstrap_summary(
                target_array,
                control,
                candidate,
                args.bootstrap_repeats,
                args.bootstrap_sample_size,
                args.seed + offset,
            )

    cat_blend_auc = family_metrics["catboost"]["blend_fold_rank_oof_auc"]
    lgb_blend_auc = family_metrics["lightgbm"]["blend_fold_rank_oof_auc"]
    metrics = {
        "run_id": args.run_id,
        "comparison": "catboost_family_blend_vs_lightgbm_family_blend",
        "folds": args.folds,
        "seed": args.seed,
        "blend": {
            "method": "within-fold percentile-rank weighted average",
            "exact_weight": args.exact_weight,
            "base_weight": 1 - args.exact_weight,
        },
        "shared": {
            "iterations": args.iterations,
            "early_stopping": False,
            "exact_features": exact_features,
            "base_features": base_features,
            "exact_categorical_features": exact_categorical,
            "base_categorical_features": base_categorical,
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
        "family_metrics": family_metrics,
        "catboost_blend_minus_lightgbm_blend": cat_blend_auc - lgb_blend_auc,
        "decision": {
            "catboost_blend_improves_exact": (
                family_metrics["catboost"]["blend_minus_exact"] > 0
            ),
            "lightgbm_blend_improves_exact": (
                family_metrics["lightgbm"]["blend_minus_exact"] > 0
            ),
            "catboost_blend_beats_lightgbm_blend": cat_blend_auc > lgb_blend_auc,
        },
        "paired_stratified_bootstrap": bootstrap_results,
        "fold_results": fold_results,
        "elapsed_seconds": time.perf_counter() - total_started,
    }

    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    metrics_path = args.output_dir / f"metrics_{args.run_id}_family_blends.json"
    with logger.step("Save metrics and OOF predictions"):
        pd.DataFrame(
            {
                "row_index": np.arange(row_count),
                ID_COLUMN: train[ID_COLUMN],
                "target": y,
                "fold": fold_number,
                **all_predictions,
            }
        ).to_csv(oof_path, index=False)
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        f"FINAL | LightGBM exact={family_metrics['lightgbm']['exact_fold_rank_oof_auc']:.10f}, "
        f"blend={lgb_blend_auc:.10f}, "
        f"delta={family_metrics['lightgbm']['blend_minus_exact']:+.10f}"
    )
    logger.info(
        f"FINAL | CatBoost exact={family_metrics['catboost']['exact_fold_rank_oof_auc']:.10f}, "
        f"blend={cat_blend_auc:.10f}, "
        f"delta={family_metrics['catboost']['blend_minus_exact']:+.10f}"
    )
    logger.info(
        f"FINAL | CatBoost blend minus LightGBM blend="
        f"{cat_blend_auc - lgb_blend_auc:+.10f}"
    )
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"OOF predictions: {oof_path}")
    logger.finish()


if __name__ == "__main__":
    main()
