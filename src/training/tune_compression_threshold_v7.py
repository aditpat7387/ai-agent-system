from pathlib import Path
import os
import json
import duckdb
import yaml
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_db_path(root: Path) -> str:
    cfg_path = root / "configs" / "data_sources.yaml"
    print(f"[INFO] Loading config from: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_path = cfg["storage"]["db_path"]
    print(f"[INFO] Config DB path: {db_path}")
    return db_path


def ensure_tables(con, tables):
    missing = []
    for table_name in tables:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        print(f"[INFO] Table check - {table_name}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(table_name)
    if missing:
        raise ValueError(f"Missing tables: {missing}")


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])


def pick_feature_columns(df):
    exclude = {
        "open_time",
        "close_time",
        "regime_label",
        "regime_group",
        "pred_label",
        "event_label",
        "pred_proba",
        "cal_pred_proba",
    }
    cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


def temporal_split(df, train_frac=0.7):
    df = df.sort_values("open_time").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    train_df = df.iloc[:cut].copy()
    test_df = df.iloc[cut:].copy()
    return train_df, test_df


def evaluate_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    row = {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "predicted_positive_count": int(y_pred.sum()),
        "predicted_negative_count": int((y_pred == 0).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return row


def main():
    root = project_root()
    print(f"[INFO] Current working directory: {os.getcwd()}")
    print(f"[INFO] Project root resolved to: {root}")

    out_dir = root / "artifacts" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    db_path = load_db_path(root)
    db_full_path = Path(db_path)
    if not db_full_path.is_absolute():
        db_full_path = root / db_path
    print(f"[INFO] Resolved DB path: {db_full_path}")

    con = duckdb.connect(str(db_full_path))
    ensure_tables(con, ["ethusd_predictions_calibrated_v7"])

    print("[INFO] Reading ethusd_predictions_calibrated_v7...")
    df = con.execute("SELECT * FROM ethusd_predictions_calibrated_v7").fetchdf()
    con.close()
    print(f"[INFO] Prediction rows: {len(df)}")

    required = ["open_time", "regime_label", "event_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["regime_label"] = df["regime_label"].astype(str)
    df["event_label"] = df["event_label"].astype(int)

    compression_df = df[df["regime_label"] == "compression"].copy()
    print(f"[INFO] Compression rows: {len(compression_df)}")

    if compression_df.empty:
        raise ValueError("No compression rows found")

    if compression_df["event_label"].nunique() < 2:
        raise ValueError("Compression subset does not contain both classes")

    feature_cols = pick_feature_columns(compression_df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions")

    print(f"[INFO] Using {len(feature_cols)} feature columns")
    print(f"[INFO] Feature columns: {feature_cols}")

    train_df, test_df = temporal_split(compression_df, train_frac=0.7)

    if train_df["event_label"].nunique() < 2:
        raise ValueError("Training split for compression has only one class")

    if test_df["event_label"].nunique() < 2:
        raise ValueError("Test split for compression has only one class")

    X_train = train_df[feature_cols]
    y_train = train_df["event_label"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["event_label"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    test_prob = pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_prob)

    print(f"[INFO] Compression ROC AUC: {roc_auc:.6f}")
    print(f"[INFO] Test probability min: {test_prob.min():.6f}")
    print(f"[INFO] Test probability max: {test_prob.max():.6f}")
    print(f"[INFO] Test probability mean: {test_prob.mean():.6f}")

    thresholds = np.round(np.arange(0.05, 0.51, 0.01), 2)

    threshold_rows = []
    for thr in thresholds:
        threshold_rows.append(evaluate_threshold(y_test, test_prob, thr))

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df["roc_auc"] = float(roc_auc)

    best_f1_row = threshold_df.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, True]
    ).iloc[0]

    best_recall_row = threshold_df.sort_values(
        ["recall", "precision", "threshold"],
        ascending=[False, False, True]
    ).iloc[0]

    best_precision_row = threshold_df[threshold_df["predicted_positive_count"] > 0].sort_values(
        ["precision", "recall", "threshold"],
        ascending=[False, False, True]
    ).iloc[0] if (threshold_df["predicted_positive_count"] > 0).any() else None

    best_threshold = float(best_f1_row["threshold"])
    best_pred = (test_prob >= best_threshold).astype(int)

    prediction_df = test_df[["open_time", "regime_label", "event_label"]].copy()
    prediction_df["compression_pred_proba"] = test_prob
    prediction_df["best_threshold"] = best_threshold
    prediction_df["compression_pred_label"] = best_pred

    threshold_path = out_dir / "compression_threshold_sweep_v7.csv"
    pred_path = out_dir / "compression_threshold_predictions_v7.csv"
    best_path = out_dir / "compression_threshold_best_metrics_v7.json"

    threshold_df.to_csv(threshold_path, index=False)
    prediction_df.to_csv(pred_path, index=False)

    best_payload = {
        "roc_auc": float(roc_auc),
        "best_f1_threshold": float(best_f1_row["threshold"]),
        "best_f1_metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in best_f1_row.to_dict().items()},
        "best_recall_threshold": float(best_recall_row["threshold"]),
        "best_recall_metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in best_recall_row.to_dict().items()},
        "feature_columns": feature_cols,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }

    if best_precision_row is not None:
        best_payload["best_precision_threshold"] = float(best_precision_row["threshold"])
        best_payload["best_precision_metrics"] = {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
            for k, v in best_precision_row.to_dict().items()
        }

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    print(f"[INFO] Saved: {threshold_path}")
    print(f"[INFO] Saved: {pred_path}")
    print(f"[INFO] Saved: {best_path}")

    print("\n[INFO] Best threshold by F1:")
    print(pd.DataFrame([best_f1_row]).to_string(index=False))

    print("\n[INFO] Best threshold by Recall:")
    print(pd.DataFrame([best_recall_row]).to_string(index=False))

    if best_precision_row is not None:
        print("\n[INFO] Best threshold by Precision:")
        print(pd.DataFrame([best_precision_row]).to_string(index=False))

    print("\n[INFO] Threshold sweep preview:")
    print(threshold_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()