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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


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


def compute_metrics(y_true, y_pred, y_prob):
    out = {
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
    return out


def train_one_group(df, group_name, feature_cols):
    sub = df[df["regime_group"] == group_name].copy()
    if sub.empty:
        print(f"[WARN] No rows for group: {group_name}")
        return None, None, None

    if sub["event_label"].nunique() < 2:
        print(f"[WARN] Group {group_name} has only one target class; skipping model training")
        return None, None, None

    train_df, test_df = temporal_split(sub, train_frac=0.7)

    if train_df["event_label"].nunique() < 2:
        print(f"[WARN] Train split for group {group_name} has only one target class; skipping")
        return None, None, None

    if test_df.empty:
        print(f"[WARN] Test split for group {group_name} is empty; skipping")
        return None, None, None

    X_train = train_df[feature_cols]
    y_train = train_df["event_label"].astype(int)
    X_test = test_df[feature_cols]
    y_test = test_df["event_label"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test, test_pred, test_prob)
    metrics["regime_group"] = group_name
    metrics["train_rows"] = int(len(train_df))
    metrics["test_rows"] = int(len(test_df))
    metrics["positive_rate_train"] = float(y_train.mean())
    metrics["positive_rate_test"] = float(y_test.mean())

    pred_df = test_df[["open_time", "regime_label", "event_label"]].copy()
    pred_df["regime_group"] = group_name
    pred_df["specialist_pred_proba"] = test_prob
    pred_df["specialist_pred_label"] = test_pred

    return pipe, metrics, pred_df


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
    df["regime_group"] = np.where(df["regime_label"] == "compression", "compression", "rest")

    feature_cols = pick_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions")

    print(f"[INFO] Using {len(feature_cols)} feature columns")
    print(f"[INFO] Feature columns: {feature_cols}")

    all_metrics = []
    all_preds = []

    for group_name in ["compression", "rest"]:
        print(f"\n[INFO] Training specialist model for: {group_name}")
        _, metrics, pred_df = train_one_group(df, group_name, feature_cols)

        if metrics is not None:
            all_metrics.append(metrics)
            all_preds.append(pred_df)
            print(json.dumps(metrics, indent=2))
        else:
            print(f"[WARN] No model result for group: {group_name}")

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    metrics_path = out_dir / "regime_specialist_metrics_v7.csv"
    preds_path = out_dir / "regime_specialist_predictions_v7.csv"

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    print(f"\n[INFO] Saved: {metrics_path}")
    print(f"[INFO] Saved: {preds_path}")

    if not metrics_df.empty:
        print("\n[INFO] Specialist model metrics:")
        print(metrics_df.to_string(index=False))

    if not preds_df.empty:
        print("\n[INFO] Prediction preview:")
        print(preds_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()