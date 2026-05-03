# prediction_agent.py — FINAL CORRECTED VERSION
# Reads from ethusd_features_1h_v2 (23-feature historical schema)
# Runs inference using hist_gradient_boosting_binary.joblib
# Recreates ethusd_predictions_calibrated_v7 each run to avoid schema drift

import sys
import traceback
import duckdb
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_COLS = [
    "return_1h",
    "log_return_1h",
    "return_4h",
    "return_24h",
    "volume_change_1h",
    "trades_change_1h",
    "high_low_range_pct",
    "vol_24h",
    "vol_72h",
    "atr_14_pct",
    "rsi_14",
    "close_vs_sma_7",
    "close_vs_sma_24",
    "sma_7_vs_24",
    "sma_24_vs_72",
    "rel_volume_24",
    "rel_trades_24",
    "taker_buy_ratio",
    "hour_of_day",
    "day_of_week",
    "dist_to_bb_upper_pct",
    "dist_to_bb_lower_pct",
    "regime_proxy_24h",
]

FEATURES_SOURCE_TABLE = "ethusd_features_1h_v2"


def run_prediction_agent(cfg: dict, context: dict) -> dict:
    pa_cfg = cfg["prediction_agent"]
    paths = cfg["paths"]
    tables = cfg["tables"]

    db_path = PROJECT_ROOT / paths["db_path"]
    pred_table = tables["predictions"]
    pred_window = int(pa_cfg["prediction_window_rows"])
    threshold = float(cfg["signal_agent"]["high_confidence_threshold"])

    model_path = Path(
        context.get("model_path") or
        str(PROJECT_ROOT / pa_cfg["model_path"])
    )

    con = duckdb.connect(str(db_path))

    try:
        if not model_path.exists():
            con.close()
            return {
                "status": "failed",
                "error": "Model not found: " + str(model_path),
                "rows_predicted": 0,
            }

        pipeline = joblib.load(model_path)
        print("[PRED] Loaded: " + model_path.name)

        feature_df = con.execute(
            f"SELECT * FROM {FEATURES_SOURCE_TABLE} "
            f"ORDER BY open_time DESC "
            f"LIMIT {pred_window}"
        ).df()

        if feature_df.empty:
            con.close()
            return {
                "status": "skipped",
                "reason": "feature_table_empty",
                "rows_predicted": 0,
            }

        feature_df = feature_df.sort_values("open_time").reset_index(drop=True)

        missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
        if missing:
            con.close()
            return {
                "status": "failed",
                "error": "Missing feature columns: " + str(missing),
                "rows_predicted": 0,
            }

        X = feature_df[FEATURE_COLS].copy()
        valid_mask = X.notna().all(axis=1)
        X_valid = X[valid_mask]
        df_valid = feature_df[valid_mask].copy()

        if X_valid.empty:
            con.close()
            return {
                "status": "skipped",
                "reason": "all_rows_have_nan_features",
                "rows_predicted": 0,
            }

        pred_labels = pipeline.predict(X_valid)
        pred_probas = pipeline.predict_proba(X_valid)[:, 1]

        df_valid["pred_label"] = pred_labels.astype(int)
        df_valid["pred_proba"] = pred_probas.round(8)

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        out_df = df_valid[["open_time", "open", "high", "low", "close", "volume", "pred_label", "pred_proba"]].copy()
        out_df["predicted_at"] = now

        con.execute(f"DROP TABLE IF EXISTS {pred_table}")
        con.register("out_df", out_df)
        con.execute(f"CREATE TABLE {pred_table} AS SELECT * FROM out_df")

        rows_predicted = int(len(out_df))
        compression_signals = int(((out_df["pred_proba"] >= threshold) & (out_df["pred_label"] == 1)).sum())
        avg_pred_proba = float(out_df["pred_proba"].mean())
        high_conf_count = int((out_df["pred_proba"] >= threshold).sum())

        print(
            f"[PRED] {rows_predicted} rows | "
            f"Avg prob: {round(avg_pred_proba, 4)} | "
            f"High-conf (>={threshold}): {high_conf_count} | "
            f"Signals: {compression_signals}"
        )

        con.close()
        return {
            "status": "success",
            "rows_predicted": rows_predicted,
            "compression_signals": compression_signals,
            "avg_pred_proba": round(avg_pred_proba, 4),
            "high_conf_count": high_conf_count,
            "model_used": model_path.name,
        }

    except Exception:
        con.close()
        return {
            "status": "failed",
            "error": traceback.format_exc(),
            "rows_predicted": 0,
        }