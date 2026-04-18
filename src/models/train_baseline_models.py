from pathlib import Path
import json
import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    "return_1h",
    "log_return_1h",
    "return_4h",
    "return_24h",
    "volume_change_1h",
    "trades_change_1h",
    "high_low_range_pct",
    "vol_24h",
    "vol_72h",
    "close_vs_sma_7",
    "close_vs_sma_24",
    "sma_7_vs_24",
    "sma_24_vs_72",
    "rel_volume_24",
    "rel_trades_24",
    "taker_buy_ratio",
    "regime_proxy_24h",
]

TARGET_COLUMN = "target_direction_4h"


def load_training_data():
    con = duckdb.connect(DB_PATH)
    df = con.execute(f"""
        SELECT
            open_time,
            close,
            {", ".join(FEATURE_COLUMNS)},
            {TARGET_COLUMN}
        FROM ethusd_training_1h
        WHERE {TARGET_COLUMN} IS NOT NULL
        ORDER BY open_time
    """).fetchdf()
    con.close()
    return df


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
    }


def build_logistic_model():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, FEATURE_COLUMNS)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    return model


def build_gbm_model():
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=200,
            random_state=42
        ))
    ])
    return model


def walk_forward_splits(df, train_size=2000, test_size=200, step_size=200):
    n = len(df)
    start = train_size
    while start + test_size <= n:
        train_idx = list(range(0, start))
        test_idx = list(range(start, start + test_size))
        yield train_idx, test_idx
        start += step_size


def run_walk_forward(df, model_name, model):
    all_predictions = []
    fold_metrics = []

    for fold_num, (train_idx, test_idx) in enumerate(walk_forward_splits(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN]
        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["fold"] = fold_num
        metrics["train_rows"] = len(train_df)
        metrics["test_rows"] = len(test_df)
        fold_metrics.append(metrics)

        fold_pred = pd.DataFrame({
            "open_time": test_df["open_time"].values,
            "close": test_df["close"].values,
            "model_name": model_name,
            "fold": fold_num,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_prob": y_prob,
        })
        all_predictions.append(fold_pred)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(fold_metrics)

    model.fit(df[FEATURE_COLUMNS], df[TARGET_COLUMN])

    return predictions_df, metrics_df, model


def main():
    df = load_training_data()
    df = df.dropna(subset=["target_direction_4h"]).reset_index(drop=True)

    logistic_model = build_logistic_model()
    gbm_model = build_gbm_model()

    results = {}

    for model_name, model in [
        ("logistic_regression", logistic_model),
        ("hist_gradient_boosting", gbm_model),
    ]:
        predictions_df, metrics_df, fitted_model = run_walk_forward(df, model_name, model)

        pred_path = PREDICTIONS_DIR / f"{model_name}_walk_forward_predictions.csv"
        metrics_path = REPORTS_DIR / f"{model_name}_walk_forward_metrics.csv"
        model_path = ARTIFACTS_DIR / f"{model_name}.joblib"

        predictions_df.to_csv(pred_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)
        joblib.dump(fitted_model, model_path)

        results[model_name] = {
            "prediction_file": str(pred_path),
            "metrics_file": str(metrics_path),
            "model_file": str(model_path),
            "summary_metrics": metrics_df.mean(numeric_only=True).to_dict(),
        }

    summary_path = REPORTS_DIR / "baseline_model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()