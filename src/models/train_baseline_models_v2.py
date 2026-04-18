from pathlib import Path
import json
import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


DB_PATH = "data/market.duckdb"
PREDICTIONS_DIR = Path("data/predictions")
ARTIFACTS_DIR = Path("models/artifacts")
REPORTS_DIR = Path("models/reports")

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "return_1h", "log_return_1h", "return_4h", "return_24h",
    "volume_change_1h", "trades_change_1h", "high_low_range_pct",
    "vol_24h", "vol_72h", "atr_14_pct", "rsi_14",
    "close_vs_sma_7", "close_vs_sma_24", "sma_7_vs_24", "sma_24_vs_72",
    "rel_volume_24", "rel_trades_24", "taker_buy_ratio",
    "hour_of_day", "day_of_week", "dist_to_bb_upper_pct", "dist_to_bb_lower_pct",
    "regime_proxy_24h",
]

BINARY_TARGET = "target_direction_4h"
MULTI_TARGET = "target_direction_4h_3class"


def load_training_data(table_name: str, target_col: str):
    con = duckdb.connect(DB_PATH)
    cols = ", ".join(FEATURE_COLUMNS)
    df = con.execute(f"""
        SELECT
            open_time,
            close,
            {cols},
            {target_col}
        FROM {table_name}
        WHERE {target_col} IS NOT NULL
        ORDER BY open_time
    """).fetchdf()
    con.close()
    return df


def compute_binary_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
    }


def compute_multiclass_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_logistic_model():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            solver="lbfgs"
        ))
    ])


def build_gbm_model():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        ))
    ])


def walk_forward_splits(df, train_size=2000, test_size=200, step_size=200):
    n = len(df)
    start = train_size
    while start + test_size <= n:
        train_idx = list(range(0, start))
        test_idx = list(range(start, start + test_size))
        yield train_idx, test_idx
        start += step_size


def run_walk_forward_binary(df, model_name, model):
    all_predictions = []
    fold_metrics = []

    for fold_num, (train_idx, test_idx) in enumerate(walk_forward_splits(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[BINARY_TARGET]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[BINARY_TARGET]

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_binary_metrics(y_test, y_pred, y_prob)
        metrics["fold"] = fold_num
        metrics["train_rows"] = len(train_df)
        metrics["test_rows"] = len(test_df)
        fold_metrics.append(metrics)

        fold_pred = pd.DataFrame({
            "open_time": test_df["open_time"].values,
            "close": test_df["close"].values,
            "model_name": model_name,
            "fold": fold_num,
            "task": "binary",
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_prob": y_prob,
        })
        all_predictions.append(fold_pred)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)

    model.fit(df[FEATURE_COLUMNS], df[BINARY_TARGET])

    return predictions_df, metrics_df, model


def run_walk_forward_multiclass(df, model_name, model):
    all_predictions = []
    fold_metrics = []

    for fold_num, (train_idx, test_idx) in enumerate(walk_forward_splits(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[MULTI_TARGET]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[MULTI_TARGET]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = compute_multiclass_metrics(y_test, y_pred)
        metrics["fold"] = fold_num
        metrics["train_rows"] = len(train_df)
        metrics["test_rows"] = len(test_df)
        fold_metrics.append(metrics)

        fold_pred = pd.DataFrame({
            "open_time": test_df["open_time"].values,
            "close": test_df["close"].values,
            "model_name": model_name,
            "fold": fold_num,
            "task": "multiclass",
            "y_true": y_test.values,
            "y_pred": y_pred,
        })
        all_predictions.append(fold_pred)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)

    model.fit(df[FEATURE_COLUMNS], df[MULTI_TARGET])

    return predictions_df, metrics_df, model


def save_results(name, task, predictions_df, metrics_df, model):
    pred_path = PREDICTIONS_DIR / f"{name}_{task}_walk_forward_predictions.csv"
    metrics_path = REPORTS_DIR / f"{name}_{task}_walk_forward_metrics.csv"
    model_path = ARTIFACTS_DIR / f"{name}_{task}.joblib"

    predictions_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    joblib.dump(model, model_path)

    return {
        "prediction_file": str(pred_path),
        "metrics_file": str(metrics_path),
        "model_file": str(model_path),
        "summary_metrics": metrics_df.mean(numeric_only=True).to_dict(),
    }


def main():
    binary_df = load_training_data("ethusd_training_1h_v2", BINARY_TARGET).dropna().reset_index(drop=True)
    multi_df = load_training_data("ethusd_training_1h_v2", MULTI_TARGET).dropna().reset_index(drop=True)

    results = {}

    for name, model in [
        ("logistic_regression", build_logistic_model()),
        ("hist_gradient_boosting", build_gbm_model()),
    ]:
        pred_b, met_b, fit_b = run_walk_forward_binary(binary_df, name, model)
        results[f"{name}_binary"] = save_results(name, "binary", pred_b, met_b, fit_b)

    for name, model in [
        ("logistic_regression", build_logistic_model()),
        ("hist_gradient_boosting", build_gbm_model()),
    ]:
        pred_m, met_m, fit_m = run_walk_forward_multiclass(multi_df, name, model)
        results[f"{name}_multiclass"] = save_results(name, "multiclass", pred_m, met_m, fit_m)

    summary_path = REPORTS_DIR / "baseline_model_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()