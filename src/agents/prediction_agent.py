# prediction_agent.py — v2 (hgb_compression + isotonic calibrator)
# Source table : ethusd_features_1h_v2  (23 features)
# Model        : hgb_compression_20260503_134449.joblib
# Calibrator   : compression_calibrator.joblib  (IsotonicRegression)
# Signal gate  : relative-rank score >= band_min (default 0.78 = top 22%)

import sys
import json
import traceback
import duckdb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_SOURCE_TABLE = "ethusd_features_1h_v2"


def _load_artifacts(pa_cfg: dict, context: dict):
    """Load model, calibrator, and feature column list from disk."""
    model_path = Path(
        context.get("model_path") or
        str(PROJECT_ROOT / pa_cfg["model_path"])
    )
    calibrator_path = Path(
        context.get("calibrator_path") or
        str(PROJECT_ROOT / pa_cfg.get("calibrator_path", "models/artifacts/compression_calibrator.joblib"))
    )
    feature_cols_path = Path(
        context.get("feature_cols_path") or
        str(PROJECT_ROOT / pa_cfg.get("feature_cols_path", "models/artifacts/feature_cols.json"))
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator not found: {calibrator_path}")
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"feature_cols.json not found: {feature_cols_path}")

    model      = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    feat_cols  = json.loads(feature_cols_path.read_text())["feature_cols"]

    return model, calibrator, feat_cols, model_path


def run_prediction_agent(cfg: dict, context: dict) -> dict:
    pa_cfg  = cfg["prediction_agent"]
    paths   = cfg["paths"]
    tables  = cfg["tables"]

    db_path     = PROJECT_ROOT / paths["db_path"]
    pred_table  = tables["predictions"]
    pred_window = int(pa_cfg["prediction_window_rows"])
    band_min    = float(pa_cfg.get("band_min", 0.78))

    con = duckdb.connect(str(db_path))

    try:
        model, calibrator, feat_cols, model_path = _load_artifacts(pa_cfg, context)
        print(f"[PRED] Loaded: {model_path.name}  |  Features: {len(feat_cols)}")

        feature_df = con.execute(
            f"SELECT * FROM {FEATURES_SOURCE_TABLE} "
            f"ORDER BY open_time DESC "
            f"LIMIT {pred_window}"
        ).df()

        if feature_df.empty:
            con.close()
            return {"status": "skipped", "reason": "feature_table_empty", "rows_predicted": 0}

        feature_df = feature_df.sort_values("open_time").reset_index(drop=True)

        missing = [c for c in feat_cols if c not in feature_df.columns]
        if missing:
            con.close()
            return {"status": "failed", "error": f"Missing feature columns: {missing}", "rows_predicted": 0}

        X          = feature_df[feat_cols].copy()
        valid_mask = X.notna().all(axis=1)
        X_valid    = X[valid_mask]
        df_valid   = feature_df[valid_mask].copy()

        if X_valid.empty:
            con.close()
            return {"status": "skipped", "reason": "all_rows_have_nan_features", "rows_predicted": 0}

        # ── FIX: use .values — matches training (retrain_subagent uses .values)
        # Eliminates: "X has feature names but was fitted without feature names"
        raw_probas = model.predict_proba(X_valid.values)[:, 1]
        cal_probas = calibrator.transform(raw_probas)

        # Relative rank score: position within [0,1] across this batch
        rel_scores = (
            pd.Series(cal_probas).rank(pct=True).values
            if len(cal_probas) > 1
            else np.ones(len(cal_probas), dtype=float)
        )

        pred_labels = (rel_scores >= band_min).astype(int)

        df_valid["pred_label"]   = pred_labels
        df_valid["pred_proba"]   = np.round(raw_probas, 8)
        df_valid["cal_proba"]    = np.round(cal_probas, 8)
        df_valid["rel_score"]    = np.round(rel_scores, 6)
        df_valid["predicted_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

        out_cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "pred_label", "pred_proba", "cal_proba", "rel_score", "predicted_at"
        ]
        out_df = df_valid[out_cols].copy()

        con.execute(f"DROP TABLE IF EXISTS {pred_table}")
        con.register("out_df", out_df)
        con.execute(f"CREATE TABLE {pred_table} AS SELECT * FROM out_df")

        rows_predicted      = len(out_df)
        compression_signals = int(out_df["pred_label"].sum())
        avg_raw_proba       = float(out_df["pred_proba"].mean())
        avg_cal_proba       = float(out_df["cal_proba"].mean())
        high_conf_count     = int((out_df["rel_score"] >= band_min).sum())

        print(
            f"[PRED] {rows_predicted} rows | "
            f"Avg raw: {avg_raw_proba:.4f} | "
            f"Avg cal: {avg_cal_proba:.4f} | "
            f"Signals (rel>={band_min}): {compression_signals}"
        )

        con.close()
        return {
            "status":             "success",
            "rows_predicted":     rows_predicted,
            "compression_signals": compression_signals,
            "avg_pred_proba":     round(avg_raw_proba, 4),
            "avg_cal_proba":      round(avg_cal_proba, 4),
            "high_conf_count":    high_conf_count,
            "model_used":         model_path.name,
        }

    except FileNotFoundError as e:
        con.close()
        return {"status": "failed", "error": str(e), "rows_predicted": 0}

    except Exception:
        con.close()
        return {"status": "failed", "error": traceback.format_exc(), "rows_predicted": 0}