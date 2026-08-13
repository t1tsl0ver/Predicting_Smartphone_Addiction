from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


TARGET = "addicted_label"
ID_COLUMN = "id"
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a LightGBM baseline.")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError(
            "--run-id may contain only letters, digits, dots, underscores and hyphens"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    if TARGET not in train or ID_COLUMN not in train:
        raise ValueError(f"Train must contain {ID_COLUMN!r} and {TARGET!r}")

    feature_columns = [
        column for column in train.columns if column not in (ID_COLUMN, TARGET)
    ]
    x = train[feature_columns].copy()
    y = train[TARGET]
    categorical = x.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    for column in categorical:
        x[column] = x[column].fillna("__MISSING__").astype("category")

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=args.validation_size,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        reg_lambda=5.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    print(f"Train shape: {x_train.shape}; validation shape: {x_valid.shape}")
    print(f"Categorical features: {categorical}")
    model.fit(
        x_train,
        y_train,
        categorical_feature=categorical,
        eval_X=x_valid,
        eval_y=y_valid,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    probabilities = model.predict_proba(
        x_valid, num_iteration=model.best_iteration_
    )[:, 1]
    auc = roc_auc_score(y_valid, probabilities)
    print(f"Validation ROC AUC: {auc:.10f}")
    print(f"Best iteration count: {model.best_iteration_}")

    metrics = {
        "run_id": args.run_id,
        "model": "lightgbm",
        "seed": RANDOM_SEED,
        "validation_size": args.validation_size,
        "validation_auc": auc,
        "requested_iterations": args.iterations,
        "best_iterations": model.best_iteration_,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "reg_lambda": 5.0,
    }
    (args.output_dir / f"metrics_{args.run_id}_lightgbm.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        {
            "row_index": x_valid.index,
            "target": y_valid.to_numpy(),
            "prediction": probabilities,
        }
    ).sort_values("row_index").to_csv(
        args.output_dir / f"validation_predictions_{args.run_id}_lightgbm.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": model.booster_.feature_importance(
                importance_type="gain"
            ),
        }
    ).sort_values("importance_gain", ascending=False).to_csv(
        args.output_dir / f"feature_importance_{args.run_id}_lightgbm.csv",
        index=False,
    )

    validation_metrics = model.evals_result_["valid_0"]
    auc_curve = validation_metrics.get("auc")
    if auc_curve:
        pd.DataFrame(
            {
                "iteration": range(1, len(auc_curve) + 1),
                "validation_auc": auc_curve,
            }
        ).to_csv(
            args.output_dir / f"learning_curve_{args.run_id}_lightgbm.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
