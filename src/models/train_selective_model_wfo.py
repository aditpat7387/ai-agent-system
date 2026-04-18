import duckdb
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


def load_config(path="configs/data_sources.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_metrics(y_true, y_pred, y_proba):
    out = {
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "recall": recall_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
        "roc_auc": np.nan,
    }
    if len(np.unique(y_true)) > 1:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            out["roc_auc"] = np.nan
    return out


def main():
    cfg = load_config()
    db_path = cfg["storage"]["db_path"]
    training_table = cfg["storage"].get("training_table", "ethusd_training_1h_v5")

    model_dir = Path("artifacts/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)
    df = con.execute(f"""
        SELECT *
        FROM {training_table}
        WHERE model_allowed_flag = 1
          AND event_label IS NOT NULL
        ORDER BY open_time
    """).fetchdf()
    con.close()

    if df.empty:
        raise ValueError(f"No rows available after filtering model_allowed_flag = 1 from {training_table}")

    feature_cols = [
        "return_1h",
        "log_return_1h",
        "vol_24h",
        "vol_72h",
        "atr_14_pct",
        "rsi_14",
        "rel_volume_24",
        "close_vs_sma_7",
        "close_vs_sma_24",
        "sma_7_vs_24",
        "sma_24_vs_72",
        "dist_to_bb_upper_pct",
        "dist_to_bb_lower_pct",
        "hour_of_day",
        "day_of_week",
        "regime_id",
    ]

    df = df.dropna(subset=feature_cols + ["event_label"]).copy()
    df["event_label"] = df["event_label"].astype(int)

    n = len(df)
    if n < 300:
        raise ValueError(f"Not enough samples for walk-forward validation: {n}")

    n_splits = 5
    test_size = max(100, n // 10)
    min_train_size = max(200, n // 3)

    fold_rows = []
    oof_parts = []

    for fold in range(n_splits):
        train_end = min_train_size + fold * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_start >= n or test_end <= test_start:
            break

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        if train_df["event_label"].nunique() < 2 or test_df["event_label"].nunique() < 2:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["event_label"]
        X_test = test_df[feature_cols]
        y_test = test_df["event_label"]

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=5,
            max_iter=250,
            min_samples_leaf=25,
            random_state=42
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = safe_metrics(y_test, pred, proba)
        metrics["fold"] = fold + 1
        metrics["train_start"] = train_df["open_time"].min()
        metrics["train_end"] = train_df["open_time"].max()
        metrics["test_start"] = test_df["open_time"].min()
        metrics["test_end"] = test_df["open_time"].max()
        metrics["train_n"] = len(train_df)
        metrics["test_n"] = len(test_df)
        metrics["train_pos_rate"] = float(y_train.mean())
        metrics["test_pos_rate"] = float(y_test.mean())
        fold_rows.append(metrics)

        part = test_df[["open_time", "close", "regime_label", "event_label"]].copy()
        part["fold"] = fold + 1
        part["pred_proba"] = proba
        part["pred_label"] = pred
        oof_parts.append(part)

    if not fold_rows:
        raise ValueError(
            f"No valid walk-forward folds were generated from {training_table}. "
            "Loosen filters or increase sample size."
        )

    fold_metrics_df = pd.DataFrame(fold_rows)
    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()

    summary = fold_metrics_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean(numeric_only=True).to_frame("mean").T

    final_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=250,
        min_samples_leaf=25,
        random_state=42
    )
    final_model.fit(df[feature_cols], df["event_label"])
    joblib.dump(final_model, model_dir / "ethusd_selective_model_wfo.joblib")

    con = duckdb.connect(db_path)
    con.register("fold_metrics_df", fold_metrics_df)
    con.register("oof_df", oof_df)
    con.register("summary_df", summary)

    con.execute("""
        CREATE OR REPLACE TABLE ethusd_wfo_fold_metrics_v4 AS
        SELECT * FROM fold_metrics_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_wfo_predictions_v4 AS
        SELECT * FROM oof_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE ethusd_wfo_summary_v4 AS
        SELECT * FROM summary_df
    """)
    con.close()

    print("Training table used:", training_table)
    print("Fold metrics:")
    print(fold_metrics_df.to_string(index=False))
    print("\nMean metrics:")
    print(summary.to_string(index=False))
    print("\nOOF rows:", len(oof_df))


if __name__ == "__main__":
    main()