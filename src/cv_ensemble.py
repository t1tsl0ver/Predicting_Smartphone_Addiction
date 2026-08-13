from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
ID_COLUMN = "id"
CATBOOST_WEIGHT = 0.15
LIGHTGBM_WEIGHT = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OOF validation and cache fold-averaged test predictions."
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--lightgbm-iterations", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-id", type=str, default="trial-007")
    return parser.parse_args()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def prepare_catboost(
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


def make_catboost(args: argparse.Namespace, fold_seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=args.catboost_iterations,
        depth=args.catboost_depth,
        learning_rate=0.08,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=fold_seed,
        thread_count=-1,
        verbose=200,
        allow_writing_files=False,
    )


def make_lightgbm(args: argparse.Namespace, fold_seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.lightgbm_iterations,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        reg_lambda=5.0,
        random_state=fold_seed,
        n_jobs=-1,
        verbosity=-1,
    )


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample_submission)
    if list(sample.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("Unexpected sample submission schema")
    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Test and sample submission ids differ")

    features = [column for column in train.columns if column not in (ID_COLUMN, TARGET)]
    if features != [column for column in test.columns if column != ID_COLUMN]:
        raise ValueError("Train and test feature schemas differ")
    y = train[TARGET]
    categorical = train[features].select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    x_cat, test_cat = prepare_catboost(train[features], test[features], categorical)
    x_lgb, test_lgb = prepare_lightgbm(train[features], test[features], categorical)

    row_count = len(train)
    test_count = len(test)
    cat_oof = np.full(row_count, np.nan)
    lgb_oof = np.full(row_count, np.nan)
    cat_rank_oof = np.full(row_count, np.nan)
    lgb_rank_oof = np.full(row_count, np.nan)
    fold_number = np.zeros(row_count, dtype=np.int8)
    cat_test_probability_sum = np.zeros(test_count)
    lgb_test_probability_sum = np.zeros(test_count)
    cat_test_rank_sum = np.zeros(test_count)
    lgb_test_rank_sum = np.zeros(test_count)
    fold_results: list[dict[str, float | int]] = []

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    total_start = time.perf_counter()

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(train[features], y), start=1
    ):
        fold_seed = args.seed + fold
        fold_start = time.perf_counter()
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        print(
            f"\n=== Fold {fold}/{args.folds}: "
            f"train={len(train_indices)}, valid={len(valid_indices)} ===",
            flush=True,
        )

        cat_model = make_catboost(args, fold_seed)
        cat_model.fit(
            x_cat.iloc[train_indices],
            y_train,
            cat_features=categorical,
            eval_set=(x_cat.iloc[valid_indices], y_valid),
            early_stopping_rounds=60,
            use_best_model=True,
        )
        cat_valid = cat_model.predict_proba(x_cat.iloc[valid_indices])[:, 1]
        cat_train = cat_model.predict_proba(x_cat.iloc[train_indices])[:, 1]
        cat_test = cat_model.predict_proba(test_cat)[:, 1]

        lgb_model = make_lightgbm(args, fold_seed)
        lgb_model.fit(
            x_lgb.iloc[train_indices],
            y_train,
            categorical_feature=categorical,
            eval_X=x_lgb.iloc[valid_indices],
            eval_y=y_valid,
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=True),
                lgb.log_evaluation(period=300),
            ],
        )
        lgb_valid = lgb_model.predict_proba(
            x_lgb.iloc[valid_indices], num_iteration=lgb_model.best_iteration_
        )[:, 1]
        lgb_train = lgb_model.predict_proba(
            x_lgb.iloc[train_indices], num_iteration=lgb_model.best_iteration_
        )[:, 1]
        lgb_test = lgb_model.predict_proba(
            test_lgb, num_iteration=lgb_model.best_iteration_
        )[:, 1]

        cat_valid_rank = percentile_rank(cat_valid)
        lgb_valid_rank = percentile_rank(lgb_valid)
        blend_valid = (
            CATBOOST_WEIGHT * cat_valid_rank + LIGHTGBM_WEIGHT * lgb_valid_rank
        )
        cat_auc = roc_auc_score(y_valid, cat_valid)
        lgb_auc = roc_auc_score(y_valid, lgb_valid)
        blend_auc = roc_auc_score(y_valid, blend_valid)
        result = {
            "fold": fold,
            "train_rows": len(train_indices),
            "validation_rows": len(valid_indices),
            "catboost_best_iterations": cat_model.get_best_iteration() + 1,
            "lightgbm_best_iterations": lgb_model.best_iteration_,
            "catboost_train_auc": roc_auc_score(y_train, cat_train),
            "catboost_validation_auc": cat_auc,
            "lightgbm_train_auc": roc_auc_score(y_train, lgb_train),
            "lightgbm_validation_auc": lgb_auc,
            "blend_validation_auc": blend_auc,
            "blend_gain_vs_lightgbm": blend_auc - lgb_auc,
            "prediction_spearman": pd.Series(cat_valid).corr(
                pd.Series(lgb_valid), method="spearman"
            ),
            "elapsed_seconds": time.perf_counter() - fold_start,
        }
        fold_results.append(result)
        print(
            f"Fold {fold}: CatBoost={cat_auc:.10f}, "
            f"LightGBM={lgb_auc:.10f}, blend={blend_auc:.10f}, "
            f"blend vs LightGBM={blend_auc - lgb_auc:+.10f}",
            flush=True,
        )

        cat_oof[valid_indices] = cat_valid
        lgb_oof[valid_indices] = lgb_valid
        cat_rank_oof[valid_indices] = cat_valid_rank
        lgb_rank_oof[valid_indices] = lgb_valid_rank
        fold_number[valid_indices] = fold
        cat_test_probability_sum += cat_test
        lgb_test_probability_sum += lgb_test
        cat_test_rank_sum += percentile_rank(cat_test)
        lgb_test_rank_sum += percentile_rank(lgb_test)

    for name, values in (
        ("cat_oof", cat_oof),
        ("lgb_oof", lgb_oof),
        ("cat_rank_oof", cat_rank_oof),
        ("lgb_rank_oof", lgb_rank_oof),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains missing or infinite values")

    blend_oof = CATBOOST_WEIGHT * cat_rank_oof + LIGHTGBM_WEIGHT * lgb_rank_oof
    cat_raw_oof_auc = roc_auc_score(y, cat_oof)
    lgb_raw_oof_auc = roc_auc_score(y, lgb_oof)
    cat_rank_oof_auc = roc_auc_score(y, cat_rank_oof)
    lgb_rank_oof_auc = roc_auc_score(y, lgb_rank_oof)
    blend_oof_auc = roc_auc_score(y, blend_oof)
    fold_gains = np.array(
        [result["blend_gain_vs_lightgbm"] for result in fold_results]
    )
    fold_wins = int((fold_gains > 0).sum())
    blend_accepted = bool(fold_gains.mean() > 0 and fold_wins >= 2)

    cat_test_probability = cat_test_probability_sum / args.folds
    lgb_test_probability = lgb_test_probability_sum / args.folds
    cat_test_rank = cat_test_rank_sum / args.folds
    lgb_test_rank = lgb_test_rank_sum / args.folds
    blend_test_rank = CATBOOST_WEIGHT * cat_test_rank + LIGHTGBM_WEIGHT * lgb_test_rank

    summary = {
        "run_id": args.run_id,
        "folds": args.folds,
        "seed": args.seed,
        "blend_weights": {
            "catboost": CATBOOST_WEIGHT,
            "lightgbm": LIGHTGBM_WEIGHT,
        },
        "catboost": {
            "depth": args.catboost_depth,
            "requested_iterations": args.catboost_iterations,
            "raw_oof_auc": cat_raw_oof_auc,
            "fold_rank_oof_auc": cat_rank_oof_auc,
        },
        "lightgbm": {
            "requested_iterations": args.lightgbm_iterations,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 100,
            "reg_lambda": 5.0,
            "raw_oof_auc": lgb_raw_oof_auc,
            "fold_rank_oof_auc": lgb_rank_oof_auc,
        },
        "blend_fold_rank_oof_auc": blend_oof_auc,
        "blend_gain_vs_lightgbm_oof": blend_oof_auc - lgb_rank_oof_auc,
        "mean_fold_gain_vs_lightgbm": float(fold_gains.mean()),
        "blend_fold_wins": fold_wins,
        "blend_accepted": blend_accepted,
        "selection_rule": "mean fold gain > 0 and blend wins at least 2 of 3 folds",
        "selected_submission": "blend" if blend_accepted else "lightgbm",
        "fold_results": fold_results,
        "elapsed_seconds": time.perf_counter() - total_start,
    }
    metrics_path = args.output_dir / f"metrics_{args.run_id}_cv.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.DataFrame(
        {
            "row_index": train.index,
            "fold": fold_number,
            "target": y,
            "catboost_prediction": cat_oof,
            "lightgbm_prediction": lgb_oof,
            "catboost_fold_rank": cat_rank_oof,
            "lightgbm_fold_rank": lgb_rank_oof,
            "blend_prediction": blend_oof,
        }
    ).to_csv(args.output_dir / f"oof_predictions_{args.run_id}.csv", index=False)

    pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN],
            "catboost_probability": cat_test_probability,
            "lightgbm_probability": lgb_test_probability,
            "catboost_cv_rank": cat_test_rank,
            "lightgbm_cv_rank": lgb_test_rank,
            "blend_cv_rank": blend_test_rank,
        }
    ).to_csv(
        args.output_dir / f"test_predictions_{args.run_id}.csv", index=False
    )

    print("\n=== OOF summary ===", flush=True)
    print(f"CatBoost raw OOF AUC: {cat_raw_oof_auc:.10f}", flush=True)
    print(f"LightGBM raw OOF AUC: {lgb_raw_oof_auc:.10f}", flush=True)
    print(f"LightGBM fold-rank OOF AUC: {lgb_rank_oof_auc:.10f}", flush=True)
    print(f"Blend fold-rank OOF AUC: {blend_oof_auc:.10f}", flush=True)
    print(
        f"Blend gain vs LightGBM: {blend_oof_auc - lgb_rank_oof_auc:+.10f}; "
        f"fold wins: {fold_wins}/{args.folds}",
        flush=True,
    )
    print(f"Selected submission: {summary['selected_submission']}", flush=True)
    print(f"Metrics saved to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
