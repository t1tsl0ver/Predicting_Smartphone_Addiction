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
from compare_oof import DEFAULT_TEST_SIZE
from cv_drop_exact_feature import expected_fold_numbers, validate_control_artifacts
from cv_family_blends import bootstrap_summary, build_feature_variants
from cv_head_to_head import make_lightgbm
from cv_lightgbm import (
    ID_COLUMN,
    RAW_NUMERIC_FEATURES,
    TARGET,
    percentile_rank,
    prepare_lightgbm,
)
from progress_logging import LightGBMProgressCallback, RunLogger, format_duration
from trial_config import (
    TRIAL_EXACT_WEIGHT,
    TRIAL_FOLDS,
    TRIAL_ITERATIONS,
    TRIAL_LIGHTGBM_LEARNING_RATE,
    TRIAL_SEED,
)


NOTIFICATIONS_PER_SCREEN_HOUR = "notifications_per_screen_hour"
APP_OPENS_PER_SCREEN_HOUR = "app_opens_per_screen_hour"
OPENS_PER_NOTIFICATION = "app_opens_per_notification"
ENTERTAINMENT_SCREEN_SHARE = "entertainment_screen_share"
WORK_STUDY_SCREEN_SHARE = "work_study_screen_share"
WEEKEND_MINUS_DAILY_SCREEN = "weekend_minus_daily_screen"
MISSING_FEATURE_COUNT = "missing_feature_count"
MISSING_PATTERN_FREQUENCY = "missing_pattern_frequency"
MISSING_PATTERN_CATEGORY = "missing_pattern_category"
FREQUENCY_SUFFIX = "_value_frequency"
SCREENING_WINNERS_BUNDLE = "screening_winners_bundle"
NOTIFICATIONS_TARGET_ENCODING = "notifications_per_day_target_encoding"
SCREENING_WINNERS_NOTIFICATIONS_TE_BUNDLE = (
    "screening_winners_plus_notifications_target_encoding"
)

ORIGINAL_PREDICTORS = (
    *RAW_NUMERIC_FEATURES,
    "gender",
    "stress_level",
    "academic_work_impact",
)

FREQUENCY_CANDIDATES = {
    f"{source}{FREQUENCY_SUFFIX}": source for source in RAW_NUMERIC_FEATURES
}
TARGET_ENCODING_CANDIDATES = {
    NOTIFICATIONS_TARGET_ENCODING: "notifications_per_day",
}

SCREENING_WINNER_FEATURES = (
    WORK_STUDY_SCREEN_SHARE,
    ENTERTAINMENT_SCREEN_SHARE,
    "notifications_per_day_value_frequency",
    "daily_screen_time_hours_value_frequency",
)

SCREENING_WINNERS_NOTIFICATIONS_TE_FEATURES = (
    *SCREENING_WINNER_FEATURES,
    NOTIFICATIONS_TARGET_ENCODING,
)

CANDIDATE_NAMES = {
    NOTIFICATIONS_PER_SCREEN_HOUR,
    APP_OPENS_PER_SCREEN_HOUR,
    OPENS_PER_NOTIFICATION,
    ENTERTAINMENT_SCREEN_SHARE,
    WORK_STUDY_SCREEN_SHARE,
    WEEKEND_MINUS_DAILY_SCREEN,
    MISSING_FEATURE_COUNT,
    MISSING_PATTERN_FREQUENCY,
    MISSING_PATTERN_CATEGORY,
    SCREENING_WINNERS_BUNDLE,
    SCREENING_WINNERS_NOTIFICATIONS_TE_BUNDLE,
    *FREQUENCY_CANDIDATES,
    *TARGET_ENCODING_CANDIDATES,
}

FAMILY_CONTROL_COLUMNS = {
    "exact": "lightgbm_exact_fold_rank",
    "base": "lightgbm_base_fold_rank",
    "blend": "lightgbm_blend_fold_rank",
}
FEATURE_TRIAL_CONTROL_COLUMNS = {
    "exact": "candidate_exact_fold_rank",
    "base": "candidate_base_fold_rank",
    "blend": "candidate_blend_fold_rank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one feature candidate or a named feature bundle in the "
            "canonical LightGBM exact/base blend against cached control OOF "
            "predictions."
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
    parser.add_argument("--candidate", choices=sorted(CANDIDATE_NAMES), required=True)
    parser.add_argument("--folds", type=int, default=TRIAL_FOLDS)
    parser.add_argument("--seed", type=int, default=TRIAL_SEED)
    parser.add_argument("--iterations", type=int, default=TRIAL_ITERATIONS)
    parser.add_argument(
        "--learning-rate", type=float, default=TRIAL_LIGHTGBM_LEARNING_RATE
    )
    parser.add_argument("--exact-weight", type=float, default=TRIAL_EXACT_WEIGHT)
    parser.add_argument("--bootstrap-repeats", type=int, default=100)
    parser.add_argument("--bootstrap-sample-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--target-encoding-alpha", type=float, default=50.0)
    parser.add_argument("--target-encoding-inner-folds", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", type=str, required=True)
    return parser.parse_args()


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def normalized_value_frequency(source: pd.Series) -> pd.Series:
    keys = source.astype("string").fillna("__MISSING__")
    frequencies = keys.value_counts(dropna=False) / len(keys)
    return keys.map(frequencies).astype(float)


def missing_pattern_keys(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = ORIGINAL_PREDICTORS,
) -> pd.Series:
    missing = [column for column in feature_columns if column not in frame]
    if missing:
        raise ValueError(f"Cannot build missing pattern; missing columns: {missing}")
    powers = 1 << np.arange(len(feature_columns), dtype=np.uint16)
    return frame[list(feature_columns)].isna().astype(np.uint16).dot(powers)


def fold_frequency_values(
    source: pd.Series,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_keys = source.iloc[train_indices].astype("string").fillna("__MISSING__")
    valid_keys = source.iloc[valid_indices].astype("string").fillna("__MISSING__")
    frequencies = train_keys.value_counts(dropna=False) / len(train_indices)
    train_values = train_keys.map(frequencies).to_numpy(dtype=float)
    valid_values = valid_keys.map(frequencies).fillna(0.0).to_numpy(dtype=float)
    return train_values, valid_values


def target_encoding_mapping(
    source: pd.Series,
    target: pd.Series,
    indices: np.ndarray,
    alpha: float,
) -> tuple[pd.Series, float]:
    if alpha < 0:
        raise ValueError("Target-encoding alpha must be non-negative")
    keys = source.iloc[indices].astype("string").fillna("__MISSING__")
    values = target.iloc[indices].to_numpy(dtype=float)
    global_mean = float(values.mean())
    statistics = pd.DataFrame({"key": keys.to_numpy(), "target": values}).groupby(
        "key", dropna=False
    )["target"].agg(["sum", "count"])
    mapping = (statistics["sum"] + alpha * global_mean) / (
        statistics["count"] + alpha
    )
    return mapping, global_mean


def apply_target_encoding(
    source: pd.Series,
    indices: np.ndarray,
    mapping: pd.Series,
    default: float,
) -> np.ndarray:
    keys = source.iloc[indices].astype("string").fillna("__MISSING__")
    return keys.map(mapping).fillna(default).to_numpy(dtype=float)


def fold_target_encoding_values(
    source: pd.Series,
    target: pd.Series,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    alpha: float,
    inner_folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if inner_folds < 2:
        raise ValueError("Target encoding requires at least two inner folds")
    outer_target = target.iloc[train_indices]
    if int(outer_target.value_counts().min()) < inner_folds:
        raise ValueError("Not enough rows per target class for inner folds")

    train_values = np.full(len(train_indices), np.nan)
    splitter = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed
    )
    dummy = np.zeros((len(train_indices), 1), dtype=np.int8)
    for inner_train_positions, inner_valid_positions in splitter.split(
        dummy, outer_target
    ):
        mapping_indices = train_indices[inner_train_positions]
        encoded_indices = train_indices[inner_valid_positions]
        mapping, default = target_encoding_mapping(
            source, target, mapping_indices, alpha
        )
        train_values[inner_valid_positions] = apply_target_encoding(
            source, encoded_indices, mapping, default
        )

    validation_mapping, validation_default = target_encoding_mapping(
        source, target, train_indices, alpha
    )
    valid_values = apply_target_encoding(
        source, valid_indices, validation_mapping, validation_default
    )
    if not np.isfinite(train_values).all() or not np.isfinite(valid_values).all():
        raise ValueError("Target encoding produced non-finite values")
    return train_values, valid_values


def add_candidate_feature(frame: pd.DataFrame, candidate: str) -> str:
    original_features = [
        column for column in frame.columns if column not in (ID_COLUMN, TARGET)
    ]
    if candidate in frame:
        raise ValueError(f"Candidate feature already exists: {candidate}")

    if candidate == NOTIFICATIONS_PER_SCREEN_HOUR:
        frame[candidate] = safe_ratio(
            frame["notifications_per_day"], frame["daily_screen_time_hours"]
        )
    elif candidate == APP_OPENS_PER_SCREEN_HOUR:
        frame[candidate] = safe_ratio(
            frame["app_opens_per_day"], frame["daily_screen_time_hours"]
        )
    elif candidate == OPENS_PER_NOTIFICATION:
        frame[candidate] = safe_ratio(
            frame["app_opens_per_day"], frame["notifications_per_day"]
        )
    elif candidate == ENTERTAINMENT_SCREEN_SHARE:
        frame[candidate] = safe_ratio(
            frame["social_media_hours"] + frame["gaming_hours"],
            frame["daily_screen_time_hours"],
        )
    elif candidate == WORK_STUDY_SCREEN_SHARE:
        frame[candidate] = safe_ratio(
            frame["work_study_hours"], frame["daily_screen_time_hours"]
        )
    elif candidate == WEEKEND_MINUS_DAILY_SCREEN:
        frame[candidate] = (
            frame["weekend_screen_time"] - frame["daily_screen_time_hours"]
        )
    elif candidate == MISSING_FEATURE_COUNT:
        frame[candidate] = frame[original_features].isna().sum(axis=1).astype(np.int8)
    elif candidate == MISSING_PATTERN_FREQUENCY:
        frame[candidate] = normalized_value_frequency(missing_pattern_keys(frame))
    elif candidate == MISSING_PATTERN_CATEGORY:
        frame[candidate] = missing_pattern_keys(frame).astype("string")
    elif candidate in FREQUENCY_CANDIDATES:
        source = FREQUENCY_CANDIDATES[candidate]
        frame[candidate] = normalized_value_frequency(frame[source])
    elif candidate in TARGET_ENCODING_CANDIDATES:
        # The placeholder establishes a numeric schema only. Every model-facing
        # value is replaced by a leak-safe outer/inner-fold encoding in main().
        frame[candidate] = 0.0
    else:
        raise ValueError(f"Unknown candidate: {candidate}")
    return candidate


def add_candidate_features(frame: pd.DataFrame, candidate: str) -> tuple[str, ...]:
    if candidate == SCREENING_WINNERS_BUNDLE:
        candidates = SCREENING_WINNER_FEATURES
    elif candidate == SCREENING_WINNERS_NOTIFICATIONS_TE_BUNDLE:
        candidates = SCREENING_WINNERS_NOTIFICATIONS_TE_FEATURES
    else:
        candidates = (candidate,)
    return tuple(add_candidate_feature(frame, feature) for feature in candidates)


def format_candidate_distribution(available: pd.Series) -> str:
    if available.empty:
        return "no non-missing values"
    if pd.api.types.is_numeric_dtype(available):
        return (
            f"min={available.min():.8f}, median={available.median():.8f}, "
            f"max={available.max():.8f}"
        )
    top = available.astype(str).value_counts().head(3)
    return "top=" + ", ".join(f"{value}:{count}" for value, count in top.items())


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.iterations < 1 or args.log_every < 1:
        raise ValueError("Iterations and log interval must be at least 1")
    if not 0 <= args.exact_weight <= 1:
        raise ValueError("--exact-weight must be between 0 and 1")
    if args.bootstrap_repeats < 2 or args.bootstrap_sample_size < 2:
        raise ValueError("Bootstrap repeats and sample size must be at least 2")
    if args.target_encoding_alpha < 0:
        raise ValueError("--target-encoding-alpha must be non-negative")
    if args.target_encoding_inner_folds < 2:
        raise ValueError("--target-encoding-inner-folds must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )


def resolve_control_prediction_columns(control: pd.DataFrame) -> dict[str, str]:
    if set(FAMILY_CONTROL_COLUMNS.values()).issubset(control.columns):
        return FAMILY_CONTROL_COLUMNS
    if set(FEATURE_TRIAL_CONTROL_COLUMNS.values()).issubset(control.columns):
        return FEATURE_TRIAL_CONTROL_COLUMNS
    raise ValueError("Control OOF does not contain a supported LightGBM prediction set")


def validate_feature_trial_control_artifacts(
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
        *FEATURE_TRIAL_CONTROL_COLUMNS.values(),
    }
    missing = sorted(required_columns - set(control.columns))
    if missing:
        raise ValueError(f"Feature-trial control OOF is missing columns: {missing}")
    if len(control) != len(train):
        raise ValueError("Control OOF row count differs from training data")
    if not np.array_equal(control["row_index"], np.arange(len(train))):
        raise ValueError("Control OOF row order differs from training data")
    if not control[ID_COLUMN].equals(train[ID_COLUMN]):
        raise ValueError("Control OOF ids differ from training data")
    if not np.array_equal(control["target"].to_numpy(), train[TARGET].to_numpy()):
        raise ValueError("Control OOF targets differ from training data")
    if not np.array_equal(
        control["fold"].to_numpy(dtype=np.int8),
        expected_fold_numbers(train[TARGET], folds, seed),
    ):
        raise ValueError("Control OOF folds differ from the requested trial split")

    if int(control_metrics.get("folds", -1)) != folds:
        raise ValueError("Control metrics use a different fold count")
    if int(control_metrics.get("seed", -1)) != seed:
        raise ValueError("Control metrics use a different seed")
    parameters = control_metrics.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Feature-trial control metrics have no parameters")
    if int(parameters.get("iterations", -1)) != iterations:
        raise ValueError("Control metrics use a different tree budget")
    if not np.isclose(
        float(parameters.get("exact_weight", np.nan)), exact_weight
    ):
        raise ValueError("Control metrics use a different blend weight")
    if not np.isfinite(
        control[list(FEATURE_TRIAL_CONTROL_COLUMNS.values())].to_numpy()
    ).all():
        raise ValueError("Control OOF contains missing or infinite predictions")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger("cv-feature-trial", args.output_dir / "logs", args.run_id)

    with logger.step("Load train and cached LightGBM control"):
        train = pd.read_csv(args.train)
        control = pd.read_csv(args.control_oof)
        control_metrics = json.loads(args.control_metrics.read_text(encoding="utf-8"))
        if TARGET not in train or ID_COLUMN not in train:
            raise ValueError("Unexpected training schema")
        control_columns = resolve_control_prediction_columns(control)
        if control_columns == FAMILY_CONTROL_COLUMNS:
            validate_control_artifacts(
                train,
                control,
                control_metrics,
                args.folds,
                args.seed,
                args.iterations,
                args.exact_weight,
            )
        else:
            validate_feature_trial_control_artifacts(
                train,
                control,
                control_metrics,
                args.folds,
                args.seed,
                args.iterations,
                args.exact_weight,
            )

    with logger.step(f"Build candidate feature set | {args.candidate}"):
        candidate_features = add_candidate_features(train, args.candidate)
        exact_features, base_features = build_feature_variants(train)
        for candidate_feature in candidate_features:
            if (
                candidate_feature not in exact_features
                or candidate_feature not in base_features
            ):
                raise ValueError("Candidates must be present in both feature variants")
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
        y = train[TARGET]

    candidate_distributions: dict[str, dict[str, object]] = {}
    for candidate_feature in candidate_features:
        if candidate_feature in TARGET_ENCODING_CANDIDATES:
            logger.info(
                f"Candidate | name={candidate_feature}, supervised values are "
                "computed inside outer/inner folds"
            )
            candidate_distributions[candidate_feature] = {
                "source": TARGET_ENCODING_CANDIDATES[candidate_feature],
                "scope": "outer-train mapping; inner-OOF values for model train",
            }
            continue
        available = train[candidate_feature].dropna()
        logger.info(
            f"Candidate | name={candidate_feature}, available={len(available)}, "
            f"missing={train[candidate_feature].isna().mean():.2%}, "
            f"unique={available.nunique()}"
        )
        if not available.empty:
            logger.info(
                "Candidate distribution | "
                + format_candidate_distribution(available)
            )
        candidate_distributions[candidate_feature] = {
            "available": int(available.size),
            "missing_rate": float(train[candidate_feature].isna().mean()),
            "unique_nonmissing": int(available.nunique()),
        }
    logger.info(
        f"Configuration | folds={args.folds}, seed={args.seed}, "
        f"trees={args.iterations}, learning_rate={args.learning_rate}, "
        f"blend={args.exact_weight:.0%} exact + {1 - args.exact_weight:.0%} base"
    )

    row_count = len(train)
    variants = {
        "exact": (x_exact, exact_categorical, exact_features),
        "base": (x_base, base_categorical, base_features),
    }
    probabilities = {
        variant: np.full(row_count, np.nan) for variant in variants
    }
    ranks = {variant: np.full(row_count, np.nan) for variant in variants}
    blend_oof = np.full(row_count, np.nan)
    importance_sums = {
        variant: np.zeros(len(details[2])) for variant, details in variants.items()
    }
    fold_results: list[dict[str, object]] = []

    control_fold = control["fold"].to_numpy(dtype=np.int8)
    controls = {
        variant: control[column].to_numpy()
        for variant, column in control_columns.items()
    }
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
        if not np.all(control_fold[valid_indices] == fold):
            raise ValueError("Cached control fold assignment changed unexpectedly")

        durations: dict[str, float] = {}
        fold_derived_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for candidate_feature in candidate_features:
            if candidate_feature in FREQUENCY_CANDIDATES:
                fold_derived_values[candidate_feature] = fold_frequency_values(
                    train[FREQUENCY_CANDIDATES[candidate_feature]],
                    train_indices,
                    valid_indices,
                )
            elif candidate_feature == MISSING_PATTERN_FREQUENCY:
                fold_derived_values[candidate_feature] = fold_frequency_values(
                    missing_pattern_keys(train), train_indices, valid_indices
                )
            elif candidate_feature in TARGET_ENCODING_CANDIDATES:
                fold_derived_values[candidate_feature] = fold_target_encoding_values(
                    train[TARGET_ENCODING_CANDIDATES[candidate_feature]],
                    y,
                    train_indices,
                    valid_indices,
                    args.target_encoding_alpha,
                    args.target_encoding_inner_folds,
                    fold_seed,
                )
        for variant, (features, categorical, _) in variants.items():
            x_train = features.iloc[train_indices]
            x_valid = features.iloc[valid_indices]
            if fold_derived_values:
                x_train = x_train.copy()
                x_valid = x_valid.copy()
                for feature, (train_values, valid_values) in fold_derived_values.items():
                    x_train[feature] = train_values
                    x_valid[feature] = valid_values
            model = make_lightgbm(args.iterations, args.learning_rate, fold_seed)
            label = f"Fold {fold}/{args.folds} LightGBM candidate {variant}"
            started = time.perf_counter()
            with logger.step(f"{label} | Train"):
                model.fit(
                    x_train,
                    y_train,
                    categorical_feature=categorical,
                    eval_X=x_valid,
                    eval_y=y_valid,
                    eval_metric="auc",
                    callbacks=[
                        LightGBMProgressCallback(
                            logger, label, args.iterations, args.log_every
                        )
                    ],
                )
            current_probability = model.predict_proba(
                x_valid
            )[:, 1]
            current_rank = percentile_rank(current_probability)
            probabilities[variant][valid_indices] = current_probability
            ranks[variant][valid_indices] = current_rank
            importance_sums[variant] += model.booster_.feature_importance(
                importance_type="gain"
            )
            durations[f"{variant}_seconds"] = time.perf_counter() - started

        current_blend = weighted_blend(
            ranks["exact"][valid_indices],
            ranks["base"][valid_indices],
            args.exact_weight,
        )
        blend_oof[valid_indices] = current_blend
        component_results: dict[str, dict[str, float]] = {}
        for variant in ("exact", "base"):
            control_auc = roc_auc_score(y_valid, controls[variant][valid_indices])
            candidate_auc = roc_auc_score(y_valid, ranks[variant][valid_indices])
            component_results[variant] = {
                "control_auc": control_auc,
                "candidate_auc": candidate_auc,
                "candidate_minus_control": candidate_auc - control_auc,
            }
        control_blend_auc = roc_auc_score(y_valid, controls["blend"][valid_indices])
        candidate_blend_auc = roc_auc_score(y_valid, current_blend)
        fold_results.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "exact": component_results["exact"],
                "base": component_results["base"],
                "blend": {
                    "control_auc": control_blend_auc,
                    "candidate_auc": candidate_blend_auc,
                    "candidate_minus_control": candidate_blend_auc
                    - control_blend_auc,
                },
                "durations": durations,
                "elapsed_seconds": time.perf_counter() - fold_started,
            }
        )
        elapsed = time.perf_counter() - total_started
        eta = elapsed / fold * (args.folds - fold)
        logger.info(
            f"FOLD RESULT | {fold}/{args.folds} | "
            f"control blend={control_blend_auc:.10f}, "
            f"candidate={candidate_blend_auc:.10f}, "
            f"delta={candidate_blend_auc - control_blend_auc:+.10f} | "
            f"elapsed={format_duration(elapsed)}, ETA={format_duration(eta)}"
        )

    for name, values in {**probabilities, **ranks, "blend": blend_oof}.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Candidate {name} predictions are not finite")

    component_metrics: dict[str, dict[str, float]] = {}
    for variant in ("exact", "base"):
        control_auc = roc_auc_score(y, controls[variant])
        candidate_auc = roc_auc_score(y, ranks[variant])
        component_metrics[variant] = {
            "control_fold_rank_oof_auc": control_auc,
            "candidate_fold_rank_oof_auc": candidate_auc,
            "candidate_minus_control": candidate_auc - control_auc,
            "prediction_spearman": float(
                spearmanr(controls[variant], ranks[variant]).statistic
            ),
        }
    control_auc = roc_auc_score(y, controls["blend"])
    candidate_auc = roc_auc_score(y, blend_oof)
    delta = candidate_auc - control_auc
    with logger.step("Paired bootstrap | candidate blend minus control"):
        bootstrap = bootstrap_summary(
            y.to_numpy(dtype=np.int8),
            controls["blend"],
            blend_oof,
            args.bootstrap_repeats,
            args.bootstrap_sample_size,
            args.seed,
        )

    won_all_folds = all(
        float(result["blend"]["candidate_minus_control"]) > 0
        for result in fold_results
    )
    interval_positive = float(bootstrap["ci95_lower"]) > 0
    metrics = {
        "run_id": args.run_id,
        "comparison": (
            "feature_bundle_lightgbm_family_blend"
            if len(candidate_features) > 1
            else "single_feature_lightgbm_family_blend"
        ),
        "control_run_id": control_metrics.get("run_id"),
        "candidate_feature": args.candidate,
        "candidate_features": list(candidate_features),
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
            "frequency_encoding_scope": (
                "fold-train mapping; unseen validation values map to zero"
                if any(
                    feature in FREQUENCY_CANDIDATES
                    or feature == MISSING_PATTERN_FREQUENCY
                    for feature in candidate_features
                )
                else None
            ),
            "target_encoding": (
                {
                    "scope": (
                        "outer-train mapping for validation; inner-OOF mapping "
                        "for model-train rows"
                    ),
                    "alpha": args.target_encoding_alpha,
                    "inner_folds": args.target_encoding_inner_folds,
                    "unseen_value": "mapping-train global target mean",
                }
                if any(
                    feature in TARGET_ENCODING_CANDIDATES
                    for feature in candidate_features
                )
                else None
            ),
        },
        "candidate_distributions": candidate_distributions,
        "components": component_metrics,
        "blend": {
            "control_fold_rank_oof_auc": control_auc,
            "candidate_fold_rank_oof_auc": candidate_auc,
            "candidate_minus_control": delta,
            "prediction_pearson": float(
                np.corrcoef(controls["blend"], blend_oof)[0, 1]
            ),
            "prediction_spearman": float(
                spearmanr(controls["blend"], blend_oof).statistic
            ),
        },
        "decision": {
            "point_estimate_improved": delta > 0,
            "won_all_folds": won_all_folds,
            "row_bootstrap_interval_positive": interval_positive,
            "screening_accepted": delta > 0 and won_all_folds and interval_positive,
        },
        "paired_stratified_bootstrap": bootstrap,
        "fold_results": fold_results,
        "elapsed_seconds": time.perf_counter() - total_started,
    }

    metrics_path = args.output_dir / f"metrics_{args.run_id}_feature.json"
    oof_path = args.output_dir / f"oof_predictions_{args.run_id}.csv"
    importance_path = args.output_dir / f"feature_importance_{args.run_id}.csv"
    with logger.step("Save metrics, OOF predictions and feature importance"):
        pd.DataFrame(
            {
                "row_index": np.arange(row_count),
                ID_COLUMN: train[ID_COLUMN],
                "target": y,
                "fold": control_fold,
                "control_blend_fold_rank": controls["blend"],
                "candidate_exact_probability": probabilities["exact"],
                "candidate_base_probability": probabilities["base"],
                "candidate_exact_fold_rank": ranks["exact"],
                "candidate_base_fold_rank": ranks["base"],
                "candidate_blend_fold_rank": blend_oof,
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
        pd.DataFrame(importance_rows).sort_values(
            ["variant", "mean_gain"], ascending=[True, False]
        ).to_csv(importance_path, index=False)
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info(
        f"FINAL | control={control_auc:.10f}, candidate={candidate_auc:.10f}, "
        f"delta={delta:+.10f}, won_all_folds={won_all_folds}, "
        f"bootstrap_positive={interval_positive}"
    )
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"OOF predictions: {oof_path}")
    logger.info(f"Feature importance: {importance_path}")
    logger.finish()


if __name__ == "__main__":
    main()
