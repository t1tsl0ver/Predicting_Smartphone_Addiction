from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from compare_oof import DEFAULT_TEST_SIZE, paired_stratified_bootstrap_deltas
from cv_ensemble import prepare_catboost
from cv_family_blends import build_feature_variants
from cv_head_to_head import DROP_FEATURES
from cv_lightgbm import ID_COLUMN, TARGET, percentile_rank
from progress_logging import CatBoostProgressCallback, RunLogger, format_duration


EXPERIMENT_FOLDS = 5
EXPERIMENT_SEED = 2026
EXPERIMENT_ITERATIONS = 3000
EXPERIMENT_LEARNING_RATE = 0.01
EXPERIMENT_DEPTH = 8
EXPERIMENT_L2_LEAF_REG = 5.0
EXPERIMENT_EARLY_STOPPING_ROUNDS = 400

DROPPED_EXACT_FEATURE = "gaming_hours_exact"
CONTROL_PREDICTION_COLUMN = "rank_blend_prediction"
PREVIOUS_CATBOOST_PREDICTION_COLUMN = "catboost_fold_rank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full five-fold exact-value CatBoost experiment and compare "
            "it with the cached best LightGBM blend."
        )
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument(
        "--control-oof",
        type=Path,
        default=Path("artifacts/oof_predictions_exp-005.csv"),
    )
    parser.add_argument(
        "--previous-catboost-oof",
        type=Path,
        default=Path("artifacts/oof_predictions_trial-022.csv"),
    )
    parser.add_argument("--folds", type=int, default=EXPERIMENT_FOLDS)
    parser.add_argument("--seed", type=int, default=EXPERIMENT_SEED)
    parser.add_argument("--iterations", type=int, default=EXPERIMENT_ITERATIONS)
    parser.add_argument(
        "--learning-rate", type=float, default=EXPERIMENT_LEARNING_RATE
    )
    parser.add_argument("--depth", type=int, default=EXPERIMENT_DEPTH)
    parser.add_argument(
        "--l2-leaf-reg", type=float, default=EXPERIMENT_L2_LEAF_REG
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=EXPERIMENT_EARLY_STOPPING_ROUNDS,
    )
    parser.add_argument("--random-strength", type=float, default=1.0)
    parser.add_argument("--border-count", type=int, default=254)
    parser.add_argument("--max-ctr-complexity", type=int, default=1)
    parser.add_argument("--leaf-estimation-iterations", type=int, default=10)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--bootstrap-repeats", type=int, default=300)
    parser.add_argument(
        "--bootstrap-sample-size", type=int, default=DEFAULT_TEST_SIZE
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--snapshot-interval", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    parser.add_argument("--run-id", default="exp-006")
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore completed fold checkpoints and retrain every fold.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if args.early_stopping_rounds < 1:
        raise ValueError("--early-stopping-rounds must be at least 1")
    if args.snapshot_interval < 1:
        raise ValueError("--snapshot-interval must be at least 1")
    if args.depth < 1 or args.l2_leaf_reg < 0:
        raise ValueError("Depth must be positive and L2 regularization non-negative")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if not 0 < args.subsample <= 1:
        raise ValueError("--subsample must be in (0, 1]")
    if args.bootstrap_repeats < 2 or args.bootstrap_sample_size < 2:
        raise ValueError("Bootstrap repeats and sample size must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )


def build_catboost_features(frame: pd.DataFrame) -> list[str]:
    exact_features, _ = build_feature_variants(frame)
    if DROPPED_EXACT_FEATURE not in exact_features:
        raise ValueError(f"Missing accepted ablation feature: {DROPPED_EXACT_FEATURE}")
    features = [
        feature for feature in exact_features if feature != DROPPED_EXACT_FEATURE
    ]
    if len(features) != len(exact_features) - 1:
        raise ValueError("Exactly one exact-value feature must be removed")
    return features


def make_full_catboost(
    args: argparse.Namespace,
    fold_seed: int,
    train_dir: Path | None = None,
) -> CatBoostClassifier:
    parameters: dict[str, Any] = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "border_count": args.border_count,
        "max_ctr_complexity": args.max_ctr_complexity,
        "leaf_estimation_iterations": args.leaf_estimation_iterations,
        "bootstrap_type": "MVS",
        "subsample": args.subsample,
        "boosting_type": "Plain",
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": fold_seed,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": train_dir is not None,
    }
    if train_dir is not None:
        train_dir.mkdir(parents=True, exist_ok=True)
        parameters.update(
            {
                "train_dir": str(train_dir),
                "save_snapshot": True,
                "snapshot_file": "training.cbsnapshot",
                "snapshot_interval": args.snapshot_interval,
            }
        )
    return CatBoostClassifier(**parameters)


def expected_fold_numbers(target: pd.Series, folds: int, seed: int) -> np.ndarray:
    result = np.zeros(len(target), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    dummy = np.zeros((len(target), 1), dtype=np.int8)
    for fold, (_, valid_indices) in enumerate(splitter.split(dummy, target), start=1):
        result[valid_indices] = fold
    return result


def load_control_prediction(
    path: Path,
    train: pd.DataFrame,
    expected_folds: np.ndarray,
    prediction_column: str,
) -> np.ndarray:
    control = pd.read_csv(path)
    required = {"row_index", "target", "fold", prediction_column}
    missing = sorted(required - set(control.columns))
    if missing:
        raise ValueError(f"Control {path} is missing columns: {missing}")
    if len(control) != len(train):
        raise ValueError(f"Control {path} row count differs from training data")
    if not np.array_equal(control["row_index"], np.arange(len(train))):
        raise ValueError(f"Control {path} row order differs from training data")
    if not np.array_equal(control["target"], train[TARGET]):
        raise ValueError(f"Control {path} targets differ from training data")
    if not np.array_equal(control["fold"].to_numpy(dtype=np.int8), expected_folds):
        raise ValueError(f"Control {path} uses different CV folds")
    values = control[prediction_column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Control {path} contains non-finite predictions")
    return values


def _series_hash(series: pd.Series) -> str:
    values = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def experiment_signature(
    args: argparse.Namespace,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> str:
    payload = {
        "folds": args.folds,
        "seed": args.seed,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "early_stopping_rounds": args.early_stopping_rounds,
        "random_strength": args.random_strength,
        "border_count": args.border_count,
        "max_ctr_complexity": args.max_ctr_complexity,
        "leaf_estimation_iterations": args.leaf_estimation_iterations,
        "subsample": args.subsample,
        "features": features,
        "categorical": categorical,
        "train_ids": _series_hash(train[ID_COLUMN]),
        "target": _series_hash(train[TARGET]),
        "test_ids": _series_hash(test[ID_COLUMN]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_fold_checkpoint(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        signature=np.asarray(result["signature"]),
        fold=np.asarray(result["fold"], dtype=np.int16),
        valid_indices=np.asarray(result["valid_indices"], dtype=np.int64),
        valid_probability=np.asarray(result["valid_probability"], dtype=np.float64),
        test_probability=np.asarray(result["test_probability"], dtype=np.float64),
        feature_importance=np.asarray(result["feature_importance"], dtype=np.float64),
        best_iteration=np.asarray(result["best_iteration"], dtype=np.int32),
        trained_iterations=np.asarray(result["trained_iterations"], dtype=np.int32),
        tree_count=np.asarray(result["tree_count"], dtype=np.int32),
        validation_auc=np.asarray(result["validation_auc"], dtype=np.float64),
        elapsed_seconds=np.asarray(result["elapsed_seconds"], dtype=np.float64),
    )
    temporary.replace(path)


def load_fold_checkpoint(
    path: Path,
    signature: str,
    fold: int,
    valid_indices: np.ndarray,
    test_count: int,
    feature_count: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            result = {
                "signature": str(data["signature"].item()),
                "fold": int(data["fold"].item()),
                "valid_indices": data["valid_indices"].astype(np.int64),
                "valid_probability": data["valid_probability"].astype(np.float64),
                "test_probability": data["test_probability"].astype(np.float64),
                "feature_importance": data["feature_importance"].astype(np.float64),
                "best_iteration": int(data["best_iteration"].item()),
                "trained_iterations": int(data["trained_iterations"].item()),
                "tree_count": int(data["tree_count"].item()),
                "validation_auc": float(data["validation_auc"].item()),
                "elapsed_seconds": float(data["elapsed_seconds"].item()),
            }
    except (OSError, ValueError, KeyError):
        return None

    arrays = [
        result["valid_probability"],
        result["test_probability"],
        result["feature_importance"],
    ]
    valid = (
        result["signature"] == signature
        and result["fold"] == fold
        and np.array_equal(result["valid_indices"], valid_indices)
        and len(result["valid_probability"]) == len(valid_indices)
        and len(result["test_probability"]) == test_count
        and len(result["feature_importance"]) == feature_count
        and all(np.isfinite(values).all() for values in arrays)
        and result["best_iteration"] >= 1
        and result["trained_iterations"] >= result["best_iteration"]
        and result["tree_count"] >= 1
        and np.isfinite(result["validation_auc"])
    )
    return result if valid else None


def bootstrap_summary(
    target: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    repeats: int,
    sample_size: int,
    seed: int,
) -> dict[str, float | int | str]:
    deltas = paired_stratified_bootstrap_deltas(
        target, control, candidate, repeats, sample_size, seed
    )
    return {
        "scope": "row-sampling uncertainty only; not training or LB-split uncertainty",
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
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("cv-catboost", args.output_dir / "logs", args.run_id)

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
        if set(train[TARGET].unique()) != {0, 1}:
            raise ValueError(f"{TARGET!r} must contain exactly 0 and 1")

    with logger.step("Build accepted CatBoost feature set"):
        features = build_catboost_features(train)
        test_features = build_catboost_features(test)
        if features != test_features:
            raise ValueError("Train and test feature schemas differ")
        categorical = train[features].select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()
        x, x_test = prepare_catboost(train[features], test[features], categorical)
        y = train[TARGET]
        fold_number = expected_fold_numbers(y, args.folds, args.seed)
        signature = experiment_signature(
            args, train, test, features, categorical
        )

    with logger.step("Load aligned cached controls"):
        control_prediction = load_control_prediction(
            args.control_oof,
            train,
            fold_number,
            CONTROL_PREDICTION_COLUMN,
        )
        previous_catboost_prediction = load_control_prediction(
            args.previous_catboost_oof,
            train,
            fold_number,
            PREVIOUS_CATBOOST_PREDICTION_COLUMN,
        )

    logger.info(
        f"Data ready | train={len(train)}, test={len(test)}, features={len(features)}, "
        f"categorical={len(categorical)}, dropped={DROPPED_EXACT_FEATURE}, "
        f"positive_rate={y.mean():.6f}"
    )
    logger.info(
        f"Full CV | folds={args.folds}, shuffle=True, seed={args.seed}, "
        f"max_iterations={args.iterations}, learning_rate={args.learning_rate}, "
        f"early_stopping_rounds={args.early_stopping_rounds}, use_best_model=True"
    )
    logger.info(
        f"CatBoost | depth={args.depth}, l2_leaf_reg={args.l2_leaf_reg}, "
        f"bootstrap=MVS, subsample={args.subsample}, random_strength={args.random_strength}, "
        f"border_count={args.border_count}, max_ctr_complexity={args.max_ctr_complexity}, "
        f"leaf_estimation_iterations={args.leaf_estimation_iterations}"
    )
    logger.info(
        f"Resume | enabled={not args.no_resume}, signature={signature[:16]}, "
        f"snapshot_interval={args.snapshot_interval}s"
    )

    row_count = len(train)
    test_count = len(test)
    raw_oof = np.full(row_count, np.nan)
    rank_oof = np.full(row_count, np.nan)
    test_probability_sum = np.zeros(test_count)
    test_rank_sum = np.zeros(test_count)
    importance_sum = np.zeros(len(features))
    fold_results: list[dict[str, Any]] = []
    checkpoint_root = args.checkpoint_dir / args.run_id / signature[:16]
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    total_started = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(x, y), start=1
    ):
        fold_seed = args.seed + fold
        checkpoint_path = checkpoint_root / f"fold-{fold}.npz"
        checkpoint = None
        if not args.no_resume:
            checkpoint = load_fold_checkpoint(
                checkpoint_path,
                signature,
                fold,
                valid_indices,
                test_count,
                len(features),
            )

        if checkpoint is not None:
            logger.info(
                f"FOLD RESUME | {fold}/{args.folds} | checkpoint={checkpoint_path} | "
                f"best_iteration={checkpoint['best_iteration']}"
            )
            result = checkpoint
            resumed = True
        else:
            fold_started = time.perf_counter()
            logger.info(
                f"FOLD START | {fold}/{args.folds} | train={len(train_indices)}, "
                f"valid={len(valid_indices)}, seed={fold_seed}"
            )
            model = make_full_catboost(
                args,
                fold_seed,
                checkpoint_root / f"catboost-info-fold-{fold}",
            )
            with logger.step(f"Fold {fold}/{args.folds} | Train CatBoost"):
                model.fit(
                    x.iloc[train_indices],
                    y.iloc[train_indices],
                    cat_features=categorical,
                    eval_set=(x.iloc[valid_indices], y.iloc[valid_indices]),
                    use_best_model=True,
                    early_stopping_rounds=args.early_stopping_rounds,
                    callbacks=[
                        CatBoostProgressCallback(
                            logger,
                            f"Fold {fold}/{args.folds} CatBoost",
                            args.iterations,
                            args.log_every,
                        )
                    ],
                )
            with logger.step(f"Fold {fold}/{args.folds} | Predictions"):
                valid_probability = model.predict_proba(x.iloc[valid_indices])[:, 1]
                test_probability = model.predict_proba(x_test)[:, 1]
                importance = model.get_feature_importance()
            validation_auc = roc_auc_score(y.iloc[valid_indices], valid_probability)
            best_iteration = model.get_best_iteration() + 1
            tree_count = int(model.tree_count_)
            evals_result = model.get_evals_result()
            validation_metrics = evals_result.get("validation", {})
            trained_iterations = max(
                (len(values) for values in validation_metrics.values()),
                default=tree_count,
            )
            result = {
                "signature": signature,
                "fold": fold,
                "valid_indices": valid_indices,
                "valid_probability": valid_probability,
                "test_probability": test_probability,
                "feature_importance": importance,
                "best_iteration": best_iteration,
                "trained_iterations": trained_iterations,
                "tree_count": tree_count,
                "validation_auc": validation_auc,
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
            save_fold_checkpoint(checkpoint_path, result)
            resumed = False
            logger.info(f"FOLD CHECKPOINT | {fold}/{args.folds} | {checkpoint_path}")

        valid_probability = result["valid_probability"]
        test_probability = result["test_probability"]
        valid_rank = percentile_rank(valid_probability)
        test_rank = percentile_rank(test_probability)
        raw_oof[valid_indices] = valid_probability
        rank_oof[valid_indices] = valid_rank
        test_probability_sum += test_probability
        test_rank_sum += test_rank
        importance_sum += result["feature_importance"]
        fold_results.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "best_iteration": result["best_iteration"],
                "trained_iterations": result["trained_iterations"],
                "tree_count": result["tree_count"],
                "validation_auc": result["validation_auc"],
                "resumed_from_checkpoint": resumed,
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        completed = len(fold_results)
        current_elapsed = time.perf_counter() - total_started
        remaining = current_elapsed / completed * (args.folds - completed)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | AUC={result['validation_auc']:.10f}, "
            f"best={result['best_iteration']}, trained={result['trained_iterations']}"
        )
        logger.info(
            f"FOLD DONE | {fold}/{args.folds} ({100 * fold / args.folds:.1f}%) | "
            f"current_run_elapsed={format_duration(current_elapsed)} | "
            f"naive_ETA={format_duration(remaining)}"
        )

    for name, values in {"raw OOF": raw_oof, "fold-rank OOF": rank_oof}.items():
        if not np.isfinite(values).all():
            raise ValueError(f"CatBoost {name} contains missing or infinite values")

    raw_auc = roc_auc_score(y, raw_oof)
    rank_auc = roc_auc_score(y, rank_oof)
    if rank_auc > raw_auc:
        selected_prediction = "fold_rank"
        selected_oof = rank_oof
        submission_values = test_rank_sum / args.folds
    else:
        selected_prediction = "raw_probability"
        selected_oof = raw_oof
        submission_values = test_probability_sum / args.folds

    control_auc = roc_auc_score(y, control_prediction)
    previous_catboost_auc = roc_auc_score(y, previous_catboost_prediction)
    selected_auc = roc_auc_score(y, selected_oof)
    logger.info("START | Paired stratified bootstrap comparisons")
    bootstrap_started = time.perf_counter()
    versus_control = bootstrap_summary(
        y.to_numpy(dtype=np.int8),
        control_prediction,
        selected_oof,
        args.bootstrap_repeats,
        args.bootstrap_sample_size,
        args.seed,
    )
    versus_previous_catboost = bootstrap_summary(
        y.to_numpy(dtype=np.int8),
        previous_catboost_prediction,
        selected_oof,
        args.bootstrap_repeats,
        args.bootstrap_sample_size,
        args.seed + 1,
    )
    logger.info(
        "DONE  | Paired stratified bootstrap comparisons | "
        f"duration={format_duration(time.perf_counter() - bootstrap_started)}"
    )

    for fold_result in fold_results:
        fold = int(fold_result["fold"])
        fold_mask = fold_number == fold
        control_fold_auc = roc_auc_score(y[fold_mask], control_prediction[fold_mask])
        fold_result["exp_005_validation_auc"] = control_fold_auc
        fold_result["catboost_minus_exp_005"] = (
            float(fold_result["validation_auc"]) - control_fold_auc
        )

    elapsed_seconds = time.perf_counter() - total_started
    parameters = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "early_stopping_rounds": args.early_stopping_rounds,
        "use_best_model": True,
        "random_strength": args.random_strength,
        "border_count": args.border_count,
        "max_ctr_complexity": args.max_ctr_complexity,
        "leaf_estimation_iterations": args.leaf_estimation_iterations,
        "bootstrap_type": "MVS",
        "subsample": args.subsample,
        "boosting_type": "Plain",
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "drop_features": list(DROP_FEATURES),
        "dropped_exact_feature": DROPPED_EXACT_FEATURE,
        "features": features,
        "categorical_features": categorical,
    }
    metrics = {
        "run_id": args.run_id,
        "model": "catboost",
        "experiment_analogue": "exp-003 full low-learning-rate CV regime",
        "folds": args.folds,
        "seed": args.seed,
        "parameters": parameters,
        "raw_oof_auc": raw_auc,
        "fold_rank_oof_auc": rank_auc,
        "selected_prediction": selected_prediction,
        "selected_oof_auc": selected_auc,
        "selection_rule": "higher OOF AUC between raw probabilities and fold ranks",
        "comparisons": {
            "exp_005_rank_blend": {
                "control_oof_auc": control_auc,
                "catboost_minus_control": selected_auc - control_auc,
                "pearson_correlation": float(
                    np.corrcoef(control_prediction, selected_oof)[0, 1]
                ),
                "spearman_correlation": float(
                    spearmanr(control_prediction, selected_oof).statistic
                ),
                "paired_stratified_bootstrap": versus_control,
            },
            "trial_022_catboost_200_trees": {
                "control_oof_auc": previous_catboost_auc,
                "catboost_minus_control": selected_auc - previous_catboost_auc,
                "paired_stratified_bootstrap": versus_previous_catboost,
            },
        },
        "fold_results": fold_results,
        "checkpoint_signature": signature,
        "current_run_elapsed_seconds": elapsed_seconds,
        "sum_original_fold_elapsed_seconds": float(
            sum(float(result["elapsed_seconds"]) for result in fold_results)
        ),
    }

    metrics_path = args.output_dir / f"metrics_{args.run_id}_cv.json"
    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    test_path = args.output_dir / f"test_predictions_{args.run_id}.csv"
    importance_path = args.output_dir / f"feature_importance_{args.run_id}.csv"
    with logger.step("Save experiment artifacts and submission"):
        pd.DataFrame(
            {
                "row_index": np.arange(row_count),
                ID_COLUMN: train[ID_COLUMN],
                "target": y,
                "fold": fold_number,
                "catboost_probability": raw_oof,
                "catboost_fold_rank": rank_oof,
            }
        ).to_csv(oof_path, index=False)
        pd.DataFrame(
            {
                ID_COLUMN: test[ID_COLUMN],
                "catboost_probability": test_probability_sum / args.folds,
                "catboost_cv_rank": test_rank_sum / args.folds,
            }
        ).to_csv(test_path, index=False)
        pd.DataFrame(
            {
                "feature": features,
                "mean_importance": importance_sum / args.folds,
            }
        ).sort_values("mean_importance", ascending=False).to_csv(
            importance_path, index=False
        )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
        args.submission.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        f"FINAL | CatBoost raw={raw_auc:.10f}, fold-rank={rank_auc:.10f}, "
        f"selected={selected_prediction} ({selected_auc:.10f})"
    )
    logger.info(
        f"FINAL | exp-005={control_auc:.10f}, "
        f"CatBoost-exp-005={selected_auc - control_auc:+.10f}"
    )
    logger.info(
        f"FINAL | trial-022 CatBoost={previous_catboost_auc:.10f}, "
        f"full-minus-trial={selected_auc - previous_catboost_auc:+.10f}"
    )
    logger.info(f"Submission saved to {args.submission}; SHA-256={digest}")
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"OOF predictions: {oof_path}")
    logger.info(f"Test predictions: {test_path}")
    logger.info(f"Feature importance: {importance_path}")
    logger.info(f"Current run time={format_duration(elapsed_seconds)}")
    logger.finish()


if __name__ == "__main__":
    main()
