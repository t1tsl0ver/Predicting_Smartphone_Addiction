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

from blend_cv import weighted_blend
from cv_family_blends import build_feature_variants
from cv_feature_trial import (
    FREQUENCY_CANDIDATES,
    SCREENING_WINNERS_ALL_TARGET_ENCODINGS_BUNDLE,
    TARGET_ENCODING_CANDIDATES,
    TRIAL_065_WITHOUT_NUMERIC_SURPRISAL_BUNDLE,
    add_candidate_features,
    apply_target_encoding,
    fold_frequency_values,
    fold_target_encoding_values,
    target_encoding_mapping,
)
from cv_head_to_head import make_lightgbm
from cv_lightgbm import ID_COLUMN, TARGET, percentile_rank, prepare_lightgbm
from progress_logging import LightGBMProgressCallback, RunLogger, format_duration


TARGET_ENCODED_BUNDLE_MODE = "trial-045-target-encoded"
TRIAL_066_NO_TE_BUNDLE_MODE = "trial-066-no-te"
FEATURE_BUNDLE_CANDIDATES = {
    TARGET_ENCODED_BUNDLE_MODE: SCREENING_WINNERS_ALL_TARGET_ENCODINGS_BUNDLE,
    TRIAL_066_NO_TE_BUNDLE_MODE: TRIAL_065_WITHOUT_NUMERIC_SURPRISAL_BUNDLE,
}
FEATURE_BUNDLE_SOURCES = {
    TARGET_ENCODED_BUNDLE_MODE: "full trial-045 bundle",
    TRIAL_066_NO_TE_BUNDLE_MODE: "accepted seven-feature trial-066 bundle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full exact/base LightGBM blend with an accepted feature "
            "bundle and optional leak-safe target encodings."
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
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--early-stopping-rounds", type=int, default=400)
    parser.add_argument("--exact-weight", type=float, default=0.60)
    parser.add_argument("--target-encoding-alpha", type=float, default=50.0)
    parser.add_argument("--target-encoding-inner-folds", type=int, default=5)
    parser.add_argument(
        "--feature-bundle",
        choices=sorted(FEATURE_BUNDLE_CANDIDATES),
        default=TARGET_ENCODED_BUNDLE_MODE,
        help="Select the historical TE bundle or the current no-TE baseline.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="exp-007")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if args.early_stopping_rounds < 0:
        raise ValueError("--early-stopping-rounds must be non-negative")
    if not 0 <= args.exact_weight <= 1:
        raise ValueError("--exact-weight must be between 0 and 1")
    if args.target_encoding_alpha < 0:
        raise ValueError("--target-encoding-alpha must be non-negative")
    if args.target_encoding_inner_folds < 2:
        raise ValueError("--target-encoding-inner-folds must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )


def frequency_values_for_test(
    train_source: pd.Series,
    test_source: pd.Series,
    train_indices: np.ndarray,
) -> np.ndarray:
    train_keys = (
        train_source.iloc[train_indices].astype("string").fillna("__MISSING__")
    )
    test_keys = test_source.astype("string").fillna("__MISSING__")
    frequencies = train_keys.value_counts(dropna=False) / len(train_indices)
    return test_keys.map(frequencies).fillna(0.0).to_numpy(dtype=float)


def build_feature_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate_bundle: str = SCREENING_WINNERS_ALL_TARGET_ENCODINGS_BUNDLE,
) -> tuple[
    tuple[str, ...],
    list[str],
    list[str],
    dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]],
]:
    train_candidates = add_candidate_features(train, candidate_bundle)
    test_candidates = add_candidate_features(test, candidate_bundle)
    if train_candidates != test_candidates:
        raise ValueError("Train and test candidate feature schemas differ")

    exact_features, base_features = build_feature_variants(train)
    test_exact_features, test_base_features = build_feature_variants(test)
    if exact_features != test_exact_features or base_features != test_base_features:
        raise ValueError("Train and test model feature schemas differ")

    exact_categorical = train[exact_features].select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    base_categorical = [
        feature for feature in exact_categorical if feature in base_features
    ]
    x_exact, test_exact = prepare_lightgbm(
        train[exact_features], test[exact_features], exact_categorical
    )
    variants = {
        "exact": (
            x_exact,
            test_exact,
            exact_categorical,
            exact_features,
        ),
        "base": (
            x_exact[base_features],
            test_exact[base_features],
            base_categorical,
            base_features,
        ),
    }
    return train_candidates, exact_features, base_features, variants


def build_fold_derived_values(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: pd.Series,
    candidate_features: tuple[str, ...],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    alpha: float,
    inner_folds: int,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for feature in candidate_features:
        if feature in FREQUENCY_CANDIDATES:
            source = FREQUENCY_CANDIDATES[feature]
            train_values, valid_values = fold_frequency_values(
                train[source], train_indices, valid_indices
            )
            test_values = frequency_values_for_test(
                train[source], test[source], train_indices
            )
            values[feature] = train_values, valid_values, test_values
        elif feature in TARGET_ENCODING_CANDIDATES:
            source = TARGET_ENCODING_CANDIDATES[feature]
            train_values, valid_values = fold_target_encoding_values(
                train[source],
                target,
                train_indices,
                valid_indices,
                alpha,
                inner_folds,
                seed,
            )
            mapping, default = target_encoding_mapping(
                train[source], target, train_indices, alpha
            )
            test_values = apply_target_encoding(
                test[source], np.arange(len(test)), mapping, default
            )
            values[feature] = train_values, valid_values, test_values
    return values


def checkpoint_signature(
    args: argparse.Namespace,
    train_rows: int,
    test_rows: int,
    exact_features: list[str],
    base_features: list[str],
) -> dict[str, object]:
    signature = {
        "run_id": args.run_id,
        "folds": args.folds,
        "seed": args.seed,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "early_stopping_rounds": args.early_stopping_rounds,
        "exact_weight": args.exact_weight,
        "target_encoding_alpha": args.target_encoding_alpha,
        "target_encoding_inner_folds": args.target_encoding_inner_folds,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "exact_features": exact_features,
        "base_features": base_features,
    }
    if args.feature_bundle != TARGET_ENCODED_BUNDLE_MODE:
        signature["feature_bundle"] = args.feature_bundle
    return signature


def prepare_checkpoint_directory(
    output_dir: Path,
    signature: dict[str, object],
) -> Path:
    directory = output_dir / "checkpoints" / str(signature["run_id"])
    directory.mkdir(parents=True, exist_ok=True)
    signature_path = directory / "signature.json"
    if signature_path.exists():
        existing = json.loads(signature_path.read_text(encoding="utf-8"))
        if existing != signature:
            raise ValueError("Existing fold checkpoints use another configuration")
    else:
        signature_path.write_text(
            json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return directory


def save_fold_checkpoint(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_name = (
        "cv-target-encoded-blend"
        if args.feature_bundle == TARGET_ENCODED_BUNDLE_MODE
        else "cv-full-feature-blend"
    )
    logger = RunLogger(process_name, args.output_dir / "logs", args.run_id)

    with logger.step("Load and validate train, test and sample submission"):
        train = pd.read_csv(args.train)
        test = pd.read_csv(args.test)
        sample = pd.read_csv(args.sample_submission)
        if TARGET not in train or ID_COLUMN not in train or ID_COLUMN not in test:
            raise ValueError("Unexpected train/test schema")
        if list(sample.columns) != [ID_COLUMN, TARGET]:
            raise ValueError("Unexpected sample submission schema")
        if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
            raise ValueError("Test and sample submission ids differ")

    candidate_bundle = FEATURE_BUNDLE_CANDIDATES[args.feature_bundle]
    feature_source = FEATURE_BUNDLE_SOURCES[args.feature_bundle]
    with logger.step(f"Build {feature_source}"):
        candidate_features, exact_features, base_features, variants = (
            build_feature_frames(train, test, candidate_bundle)
        )
        target = train[TARGET]
    active_target_encodings = tuple(
        feature for feature in candidate_features if feature in TARGET_ENCODING_CANDIDATES
    )

    signature = checkpoint_signature(
        args, len(train), len(test), exact_features, base_features
    )
    checkpoint_dir = prepare_checkpoint_directory(args.output_dir, signature)
    target_encoding_label = "disabled"
    if active_target_encodings:
        target_encoding_label = (
            f"inner-{args.target_encoding_inner_folds}-fold "
            f"alpha={args.target_encoding_alpha:g}"
        )
    logger.info(
        f"Configuration | folds={args.folds}, trees<={args.iterations}, "
        f"learning_rate={args.learning_rate}, early_stopping={args.early_stopping_rounds}, "
        f"blend={args.exact_weight:.0%}/{1 - args.exact_weight:.0%}, "
        f"target_encoding={target_encoding_label}"
    )
    logger.info(
        f"Features | exact={len(exact_features)}, base={len(base_features)}, "
        f"target_encoded={len(active_target_encodings)}"
    )

    row_count = len(train)
    test_count = len(test)
    fold_number = np.zeros(row_count, dtype=np.int8)
    probabilities = {
        variant: np.full(row_count, np.nan) for variant in variants
    }
    ranks = {variant: np.full(row_count, np.nan) for variant in variants}
    blend_oof = np.full(row_count, np.nan)
    test_probability_sums = {
        variant: np.zeros(test_count) for variant in variants
    }
    test_rank_sums = {variant: np.zeros(test_count) for variant in variants}
    test_blend_sum = np.zeros(test_count)
    importance_sums = {
        variant: np.zeros(len(details[3])) for variant, details in variants.items()
    }
    fold_results: list[dict[str, object]] = []
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    total_started = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(variants["exact"][0], target), start=1
    ):
        fold_started = time.perf_counter()
        fold_seed = args.seed + fold
        fold_number[valid_indices] = fold
        checkpoint_path = checkpoint_dir / f"fold-{fold}.npz"
        if checkpoint_path.exists():
            with np.load(checkpoint_path) as checkpoint:
                if not np.array_equal(checkpoint["valid_indices"], valid_indices):
                    raise ValueError(f"Fold {fold} checkpoint indices changed")
                for variant in variants:
                    probabilities[variant][valid_indices] = checkpoint[
                        f"{variant}_valid_probability"
                    ]
                    ranks[variant][valid_indices] = checkpoint[f"{variant}_valid_rank"]
                    test_probability_sums[variant] += checkpoint[
                        f"{variant}_test_probability"
                    ]
                    test_rank_sums[variant] += checkpoint[f"{variant}_test_rank"]
                    importance_sums[variant] += checkpoint[f"{variant}_importance"]
                blend_oof[valid_indices] = checkpoint["blend_valid"]
                test_blend_sum += checkpoint["blend_test"]
                fold_result = json.loads(str(checkpoint["fold_result"].item()))
                fold_results.append(fold_result)
            logger.info(f"FOLD {fold}/{args.folds} | restored from {checkpoint_path}")
            continue

        logger.info(
            f"FOLD START | {fold}/{args.folds} | train={len(train_indices)}, "
            f"valid={len(valid_indices)}, seed={fold_seed}"
        )
        with logger.step(f"Fold {fold}/{args.folds} | Build leak-safe encodings"):
            derived = build_fold_derived_values(
                train,
                test,
                target,
                candidate_features,
                train_indices,
                valid_indices,
                args.target_encoding_alpha,
                args.target_encoding_inner_folds,
                fold_seed,
            )

        current_test_probabilities: dict[str, np.ndarray] = {}
        current_test_ranks: dict[str, np.ndarray] = {}
        current_importances: dict[str, np.ndarray] = {}
        best_iterations: dict[str, int] = {}
        durations: dict[str, float] = {}
        for variant, (features, test_features, categorical, _) in variants.items():
            x_train = features.iloc[train_indices].copy()
            x_valid = features.iloc[valid_indices].copy()
            x_test = test_features.copy()
            for feature, (train_values, valid_values, test_values) in derived.items():
                x_train[feature] = train_values
                x_valid[feature] = valid_values
                x_test[feature] = test_values

            model = make_lightgbm(args.iterations, args.learning_rate, fold_seed)
            label = f"Fold {fold}/{args.folds} LightGBM {variant}"
            callbacks: list[object] = [
                LightGBMProgressCallback(
                    logger, label, args.iterations, args.log_every
                )
            ]
            if args.early_stopping_rounds > 0:
                callbacks.append(
                    lgb.early_stopping(args.early_stopping_rounds, verbose=False)
                )
            started = time.perf_counter()
            with logger.step(f"{label} | Train"):
                model.fit(
                    x_train,
                    target.iloc[train_indices],
                    categorical_feature=categorical,
                    eval_X=x_valid,
                    eval_y=target.iloc[valid_indices],
                    eval_metric="auc",
                    callbacks=callbacks,
                )
            best_iteration = int(model.best_iteration_ or model.n_iter_)
            valid_probability = model.predict_proba(
                x_valid, num_iteration=best_iteration
            )[:, 1]
            test_probability = model.predict_proba(
                x_test, num_iteration=best_iteration
            )[:, 1]
            valid_rank = percentile_rank(valid_probability)
            test_rank = percentile_rank(test_probability)
            probabilities[variant][valid_indices] = valid_probability
            ranks[variant][valid_indices] = valid_rank
            current_test_probabilities[variant] = test_probability
            current_test_ranks[variant] = test_rank
            test_probability_sums[variant] += test_probability
            test_rank_sums[variant] += test_rank
            importance = model.booster_.feature_importance(importance_type="gain")
            importance_sums[variant] += importance
            current_importances[variant] = importance
            best_iterations[variant] = best_iteration
            durations[f"{variant}_seconds"] = time.perf_counter() - started

        current_blend = weighted_blend(
            ranks["exact"][valid_indices],
            ranks["base"][valid_indices],
            args.exact_weight,
        )
        current_test_blend = weighted_blend(
            current_test_ranks["exact"],
            current_test_ranks["base"],
            args.exact_weight,
        )
        blend_oof[valid_indices] = current_blend
        test_blend_sum += current_test_blend
        fold_result = {
            "fold": fold,
            "train_rows": len(train_indices),
            "validation_rows": len(valid_indices),
            "best_iterations": best_iterations,
            "exact_auc": roc_auc_score(
                target.iloc[valid_indices], ranks["exact"][valid_indices]
            ),
            "base_auc": roc_auc_score(
                target.iloc[valid_indices], ranks["base"][valid_indices]
            ),
            "blend_auc": roc_auc_score(target.iloc[valid_indices], current_blend),
            "durations": durations,
            "elapsed_seconds": time.perf_counter() - fold_started,
        }
        fold_results.append(fold_result)

        checkpoint_arrays: dict[str, np.ndarray] = {
            "valid_indices": valid_indices,
            "blend_valid": current_blend,
            "blend_test": current_test_blend,
            "fold_result": np.asarray(json.dumps(fold_result)),
        }
        for variant in variants:
            checkpoint_arrays[f"{variant}_valid_probability"] = probabilities[
                variant
            ][valid_indices]
            checkpoint_arrays[f"{variant}_valid_rank"] = ranks[variant][valid_indices]
            checkpoint_arrays[f"{variant}_test_probability"] = (
                current_test_probabilities[variant]
            )
            checkpoint_arrays[f"{variant}_test_rank"] = current_test_ranks[variant]
            checkpoint_arrays[f"{variant}_importance"] = current_importances[variant]
        save_fold_checkpoint(checkpoint_path, checkpoint_arrays)
        elapsed = time.perf_counter() - total_started
        eta = elapsed / fold * (args.folds - fold)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | exact={fold_result['exact_auc']:.10f}, "
            f"base={fold_result['base_auc']:.10f}, blend={fold_result['blend_auc']:.10f} | "
            f"best_iterations={best_iterations} | elapsed={format_duration(elapsed)}, "
            f"ETA={format_duration(eta)}"
        )

    if not np.isfinite(blend_oof).all() or not all(
        np.isfinite(values).all() for values in probabilities.values()
    ):
        raise ValueError("OOF predictions contain missing or infinite values")

    component_metrics: dict[str, dict[str, float]] = {}
    for variant in variants:
        component_metrics[variant] = {
            "raw_oof_auc": roc_auc_score(target, probabilities[variant]),
            "fold_rank_oof_auc": roc_auc_score(target, ranks[variant]),
        }
    blend_auc = roc_auc_score(target, blend_oof)
    test_probabilities = {
        variant: values / args.folds
        for variant, values in test_probability_sums.items()
    }
    test_ranks = {
        variant: values / args.folds for variant, values in test_rank_sums.items()
    }
    test_blend = test_blend_sum / args.folds
    elapsed_seconds = time.perf_counter() - total_started

    target_encoding_scope = None
    if active_target_encodings:
        target_encoding_scope = (
            "outer-train mapping for validation/test; inner-OOF mapping "
            "for model-train rows"
        )
    metrics = {
        "run_id": args.run_id,
        "model": (
            "lightgbm_exact_base_target_encoded_blend"
            if active_target_encodings
            else "lightgbm_exact_base_full_feature_blend"
        ),
        "feature_bundle": args.feature_bundle,
        "feature_source": feature_source,
        "folds": args.folds,
        "seed": args.seed,
        "parameters": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "num_leaves": 31,
            "min_child_samples": 20,
            "early_stopping_rounds": args.early_stopping_rounds,
            "exact_weight": args.exact_weight,
            "base_weight": 1 - args.exact_weight,
            "target_encoding_alpha": (
                args.target_encoding_alpha if active_target_encodings else None
            ),
            "target_encoding_inner_folds": (
                args.target_encoding_inner_folds if active_target_encodings else None
            ),
            "target_encoding_scope": target_encoding_scope,
            "frequency_encoding_scope": (
                "outer-train mapping; unseen validation/test values map to zero"
            ),
            "exact_features": exact_features,
            "base_features": base_features,
        },
        "components": component_metrics,
        "blend_fold_rank_oof_auc": blend_auc,
        "fold_results": fold_results,
        "elapsed_seconds": elapsed_seconds,
    }

    metrics_suffix = (
        "target_encoded_blend" if active_target_encodings else "full_feature_blend"
    )
    metrics_path = args.output_dir / f"metrics_{args.run_id}_{metrics_suffix}.json"
    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    test_path = args.output_dir / f"test_predictions_{args.run_id}.csv"
    importance_path = args.output_dir / f"feature_importance_{args.run_id}.csv"
    digest: str | None = None
    with logger.step("Save metrics, OOF, test predictions and importance"):
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "row_index": np.arange(row_count),
                "fold": fold_number,
                "target": target,
                "exact_probability": probabilities["exact"],
                "base_probability": probabilities["base"],
                "exact_fold_rank": ranks["exact"],
                "base_fold_rank": ranks["base"],
                "lightgbm_fold_rank": blend_oof,
            }
        ).to_csv(oof_path, index=False)
        pd.DataFrame(
            {
                ID_COLUMN: test[ID_COLUMN],
                "exact_probability": test_probabilities["exact"],
                "base_probability": test_probabilities["base"],
                "exact_cv_rank": test_ranks["exact"],
                "base_cv_rank": test_ranks["base"],
                "lightgbm_cv_rank": test_blend,
            }
        ).to_csv(test_path, index=False)
        importance_rows = []
        for variant, (_, _, _, feature_names) in variants.items():
            for feature, gain in zip(
                feature_names, importance_sums[variant] / args.folds, strict=True
            ):
                importance_rows.append(
                    {"variant": variant, "feature": feature, "mean_gain": gain}
                )
        pd.DataFrame(importance_rows).sort_values(
            ["variant", "mean_gain"], ascending=[True, False]
        ).to_csv(importance_path, index=False)

        if args.submission is not None:
            submission = sample.copy()
            submission[TARGET] = test_blend
            submission.to_csv(args.submission, index=False)
            digest = hashlib.sha256(args.submission.read_bytes()).hexdigest()
            args.submission.with_suffix(".metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "selected_prediction": "exact_base_fold_rank_blend",
                        "exact_weight": args.exact_weight,
                        "base_weight": 1 - args.exact_weight,
                        "rows": len(submission),
                        "minimum_prediction": float(test_blend.min()),
                        "maximum_prediction": float(test_blend.max()),
                        "sha256": digest,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    logger.info(
        f"FINAL | exact={component_metrics['exact']['fold_rank_oof_auc']:.10f}, "
        f"base={component_metrics['base']['fold_rank_oof_auc']:.10f}, "
        f"blend={blend_auc:.10f}"
    )
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"OOF predictions: {oof_path}")
    logger.info(f"Test predictions: {test_path}")
    if args.submission is not None:
        logger.info(f"Submission: {args.submission} | SHA-256={digest}")
    logger.info(f"Total CV time={format_duration(elapsed_seconds)}")
    logger.finish()


if __name__ == "__main__":
    main()
