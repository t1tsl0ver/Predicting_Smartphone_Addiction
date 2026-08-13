from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "addicted_label"
ID_COLUMN = "id"
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a model and build a Kaggle submission."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument(
        "--model",
        choices=("logistic", "catboost"),
        default="catboost",
    )
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional safe label used in metric and diagnostic filenames.",
    )
    parser.add_argument(
        "--save-validation-predictions",
        action="store_true",
        help="Save validation row indices, targets and probabilities.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Only calculate validation AUC; do not train on all rows.",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip validation and train on all rows with --iterations trees.",
    )
    return parser.parse_args()


def check_inputs(
    train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame
) -> list[str]:
    if TARGET not in train:
        raise ValueError(f"Training data has no {TARGET!r} column")
    if TARGET in test:
        raise ValueError(f"Test data unexpectedly contains {TARGET!r}")
    if ID_COLUMN not in train or ID_COLUMN not in test:
        raise ValueError(f"Both datasets must contain {ID_COLUMN!r}")
    if list(sample_submission.columns) != [ID_COLUMN, TARGET]:
        raise ValueError(
            f"Sample submission columns must be {[ID_COLUMN, TARGET]}, "
            f"got {list(sample_submission.columns)}"
        )
    if len(test) != len(sample_submission):
        raise ValueError("Test and sample submission row counts do not match")
    if not test[ID_COLUMN].equals(sample_submission[ID_COLUMN]):
        raise ValueError("Test and sample submission ids are not in the same order")

    feature_columns = [
        column for column in train.columns if column not in (ID_COLUMN, TARGET)
    ]
    if feature_columns != [column for column in test.columns if column != ID_COLUMN]:
        raise ValueError("Train and test feature schemas do not match")
    if not set(train[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET!r} must be binary")
    return feature_columns


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def prepare_catboost_categories(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = train_features.copy()
    test_features = test_features.copy()
    for column in categorical:
        train_features[column] = train_features[column].fillna("__MISSING__").astype(str)
        test_features[column] = test_features[column].fillna("__MISSING__").astype(str)
    return train_features, test_features


def make_catboost(
    iterations: int, depth: int, verbose: int | bool
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=RANDOM_SEED,
        thread_count=-1,
        verbose=verbose,
        allow_writing_files=False,
    )


def validate_catboost(
    train_features: pd.DataFrame,
    target: pd.Series,
    categorical: list[str],
    args: argparse.Namespace,
) -> tuple[float, int, CatBoostClassifier, pd.DataFrame]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_features,
        target,
        test_size=args.validation_size,
        random_state=RANDOM_SEED,
        stratify=target,
    )
    model = make_catboost(args.iterations, args.depth, verbose=100)
    model.fit(
        x_train,
        y_train,
        cat_features=categorical,
        eval_set=(x_valid, y_valid),
        early_stopping_rounds=args.early_stopping_rounds,
        use_best_model=True,
    )
    probabilities = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, probabilities)
    best_iteration = model.get_best_iteration() + 1
    validation_predictions = pd.DataFrame(
        {
            "row_index": x_valid.index,
            "target": y_valid.to_numpy(),
            "prediction": probabilities,
        }
    ).sort_values("row_index")
    return auc, best_iteration, model, validation_predictions


def make_logistic(
    numeric: list[str], categorical: list[str]
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
        ]
    )


def validate_logistic(
    train_features: pd.DataFrame,
    target: pd.Series,
    numeric: list[str],
    categorical: list[str],
    validation_size: float,
) -> tuple[float, Pipeline]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_features,
        target,
        test_size=validation_size,
        random_state=RANDOM_SEED,
        stratify=target,
    )
    model = make_logistic(numeric, categorical)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    return roc_auc_score(y_valid, probabilities), model


def write_submission(
    sample_submission: pd.DataFrame,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    if len(probabilities) != len(sample_submission):
        raise ValueError("Prediction count does not match submission row count")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinity")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions are outside [0, 1]")

    submission = sample_submission.copy()
    submission[TARGET] = probabilities
    submission.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    if args.skip_final and args.final_only:
        raise ValueError("--skip-final and --final-only cannot be used together")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample_submission = pd.read_csv(args.sample_submission)
    features = check_inputs(train, test, sample_submission)

    x = train[features]
    x_test = test[features]
    y = train[TARGET]
    categorical = categorical_columns(x)
    numeric = [column for column in features if column not in categorical]

    print(f"Train shape: {train.shape}; test shape: {test.shape}")
    print(f"Positive rate: {y.mean():.6f}")
    print(f"Model: {args.model}")

    metrics: dict[str, float | int | str] = {
        "model": args.model,
        "seed": RANDOM_SEED,
        "validation_size": args.validation_size,
    }
    if args.run_id:
        metrics["run_id"] = args.run_id

    if args.model == "catboost":
        metrics["requested_iterations"] = args.iterations
        metrics["depth"] = args.depth
        x, x_test = prepare_catboost_categories(x, x_test, categorical)
        if args.final_only:
            best_iterations = args.iterations
            metrics["best_iterations"] = best_iterations
        else:
            auc, best_iterations, validation_model, validation_predictions = validate_catboost(
                x, y, categorical, args
            )
            metrics["validation_auc"] = auc
            metrics["best_iterations"] = best_iterations
            print(f"Validation ROC AUC: {auc:.6f}")
            print(f"Best iteration count: {best_iterations}")

            importances = pd.DataFrame(
                {
                    "feature": features,
                    "importance": validation_model.get_feature_importance(),
                }
            ).sort_values("importance", ascending=False)
            diagnostic_suffix = f"_{args.run_id}" if args.run_id else ""
            importances.to_csv(
                args.output_dir / f"feature_importance{diagnostic_suffix}.csv",
                index=False,
            )
            print("\nFeature importance:")
            print(importances.to_string(index=False))

            validation_auc = (
                validation_model.get_evals_result()
                .get("validation", {})
                .get("AUC")
            )
            if validation_auc:
                pd.DataFrame(
                    {
                        "iteration": range(1, len(validation_auc) + 1),
                        "validation_auc": validation_auc,
                    }
                ).to_csv(
                    args.output_dir / f"learning_curve{diagnostic_suffix}.csv",
                    index=False,
                )
            if args.save_validation_predictions:
                validation_predictions.to_csv(
                    args.output_dir
                    / f"validation_predictions{diagnostic_suffix}_catboost.csv",
                    index=False,
                )

        if not args.skip_final:
            print("\nTraining final model on all training rows...")
            final_model = make_catboost(best_iterations, args.depth, verbose=100)
            final_model.fit(x, y, cat_features=categorical)
            final_model.save_model(args.output_dir / "catboost_model.cbm")
            test_probabilities = final_model.predict_proba(x_test)[:, 1]
    else:
        if not args.final_only:
            auc, validation_model = validate_logistic(
                x, y, numeric, categorical, args.validation_size
            )
            metrics["validation_auc"] = auc
            print(f"Validation ROC AUC: {auc:.6f}")

        if not args.skip_final:
            print("\nTraining final model on all training rows...")
            final_model = make_logistic(numeric, categorical)
            final_model.fit(x, y)
            joblib.dump(final_model, args.output_dir / "logistic_model.joblib")
            test_probabilities = final_model.predict_proba(x_test)[:, 1]

    metrics_parts = ["metrics"]
    if args.run_id:
        metrics_parts.append(args.run_id)
    metrics_parts.append(args.model)
    if args.final_only:
        metrics_parts.append("final")
    metrics_path = args.output_dir / f"{'_'.join(metrics_parts)}.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Metrics saved to {metrics_path}")

    if not args.skip_final:
        submission_path = Path("submission.csv")
        write_submission(sample_submission, test_probabilities, submission_path)
        print(f"Submission saved to {submission_path} ({len(test_probabilities)} rows)")
        print(
            "Prediction range: "
            f"[{test_probabilities.min():.6f}, {test_probabilities.max():.6f}]"
        )


if __name__ == "__main__":
    main()
