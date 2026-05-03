# =============================================================================
# retrain_subagent.py — M9 Self-Improvement Loop
# Spawned reactively by orchestrator when drift_checker detects PF degradation.
# =============================================================================

import sys
import traceback
import duckdb
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timezone

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

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

FEATURES_TABLE   = "ethusd_features_1h_v2"
REGIME_FILTER    = "compression"
OOS_RATIO        = 0.20
MIN_TRAIN_ROWS   = 200
SIGNAL_THRESHOLD = 0.85

# Label definition — must match original v7 training
# 1 = close price rises >= LABEL_PCT_THRESHOLD within next LABEL_HORIZON hours
LABEL_HORIZON       = 4    # look-forward window in hours
LABEL_PCT_THRESHOLD = 0.01 # 1% move = positive label


def _compute_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes forward-looking binary label on sorted OHLCV dataframe.
    label = 1 if max(close[t+1 .. t+LABEL_HORIZON]) >= close[t] * (1 + LABEL_PCT_THRESHOLD)
    This is the same definition used in the original v7 compression specialist training.
    """
    df = df.sort_values("open_time").reset_index(drop=True)
    future_max = (
        df["close"]
        .shift(-1)
        .rolling(window=LABEL_HORIZON, min_periods=1)
        .max()
        .shift(-(LABEL_HORIZON - 1))
    )
    df["label"] = (
        future_max >= df["close"] * (1 + LABEL_PCT_THRESHOLD)
    ).astype(int)
    # Last LABEL_HORIZON rows have no valid future — drop them
    df = df.iloc[:-LABEL_HORIZON].copy()
    return df


def _compute_profit_factor(y_true, proba, threshold=0.60):
    mask = proba >= threshold
    if mask.sum() == 0:
        return 0.0
    wins   = int((y_true[mask] == 1).sum())
    losses = int((y_true[mask] == 0).sum())
    if losses == 0:
        return float(wins) if wins > 0 else 0.0
    return round(wins / losses, 4)


def _ensure_model_registry(con):
    con.execute("DROP TABLE IF EXISTS model_registry")
    con.execute("""
        CREATE TABLE model_registry (
            model_id    VARCHAR,
            model_path  VARCHAR,
            trained_at  TIMESTAMP,
            train_rows  INTEGER,
            oos_pf      DOUBLE,
            baseline_pf DOUBLE,
            improved    BOOLEAN,
            active      BOOLEAN
        )
    """)


def run_retrain_subagent(cfg: dict, context: dict) -> dict:

    if not context.get("drift_detected", False):
        return {"status": "skipped", "reason": "no_drift_detected"}

    paths      = cfg["paths"]
    db_path    = PROJECT_ROOT / paths["db_path"]
    models_dir = PROJECT_ROOT / paths.get("models_dir", "models/artifacts")
    baseline_pf = float(
        context.get("baseline_pf",
        cfg.get("drift_checker", {}).get("baseline_pf", 1.735))
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    try:
        _ensure_model_registry(con)

        # ── 1. Load features ──────────────────────────────────────────────────
        try:
            df = con.execute(
                f"SELECT * FROM {FEATURES_TABLE} ORDER BY open_time"
            ).df()
        except Exception as e:
            con.close()
            return {"status": "failed", "error": f"Feature table read failed: {e}"}

        if df.empty:
            con.close()
            return {"status": "failed", "error": "Feature table is empty"}

        # ── 2. Validate feature columns ───────────────────────────────────────
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            con.close()
            return {"status": "failed", "error": f"Missing feature columns: {missing}"}

        # ── 3. Compute forward-looking label (not stored in table) ────────────
        df = _compute_label(df)
        label_rate = round(df["label"].mean(), 4)
        print(f"[RETRAIN] Label computed | Positive rate: {label_rate} | Rows: {len(df)}")

        # ── 4. Filter to compression regime ───────────────────────────────────
        if "regime_proxy_24h" in df.columns:
            compression_df = df[df["regime_proxy_24h"] == 0].copy()
        else:
            compression_df = df.copy()
        print(f"[RETRAIN] Compression rows: {len(compression_df)} / {len(df)} total")

        if len(compression_df) < MIN_TRAIN_ROWS:
            con.close()
            return {
                "status": "skipped",
                "reason": f"Insufficient compression rows: {len(compression_df)} < {MIN_TRAIN_ROWS}",
            }

        # ── 5. Walk-forward OOS split (no data leakage) ───────────────────────
        compression_df = compression_df.sort_values("open_time").reset_index(drop=True)
        split_idx = int(len(compression_df) * (1 - OOS_RATIO))
        train_df  = compression_df.iloc[:split_idx].copy()
        oos_df    = compression_df.iloc[split_idx:].copy()

        print(f"[RETRAIN] Train: {len(train_df)} rows | OOS: {len(oos_df)} rows")

        X_train = train_df[FEATURE_COLS].copy()
        y_train = train_df["label"].astype(int)
        X_oos   = oos_df[FEATURE_COLS].copy()
        y_oos   = oos_df["label"].astype(int)

        train_mask = X_train.notna().all(axis=1)
        oos_mask   = X_oos.notna().all(axis=1)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_oos,   y_oos   = X_oos[oos_mask],     y_oos[oos_mask]

        if len(X_train) < MIN_TRAIN_ROWS:
            con.close()
            return {
                "status": "skipped",
                "reason": f"Train rows after NaN drop: {len(X_train)} < {MIN_TRAIN_ROWS}",
            }

        # ── 6. Retrain HGB + IsotonicCalibration ──────────────────────────────
        print("[RETRAIN] Training HistGradientBoostingClassifier...")
        base_model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )
        base_model.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        calibrated.fit(X_train, y_train)

        # ── 7. OOS evaluation ─────────────────────────────────────────────────
        if len(X_oos) == 0:
            new_pf   = 0.0
            improved = False
        else:
            oos_proba          = calibrated.predict_proba(X_oos)[:, 1]
            OOS_EVAL_THRESHOLD = 0.60
            new_pf             = _compute_profit_factor(y_oos.values, oos_proba, threshold=OOS_EVAL_THRESHOLD)
            improved           = new_pf >= baseline_pf * 0.95

        print(f"[RETRAIN] OOS PF: {new_pf} | Baseline: {baseline_pf} | Improved: {improved}")

        # ── 8. Save or reject ─────────────────────────────────────────────────
        now            = datetime.now(timezone.utc)
        model_id       = f"hgb_compression_{now.strftime('%Y%m%d_%H%M%S')}"
        new_model_path = models_dir / f"{model_id}.joblib"

        insert_values = [
            model_id, str(new_model_path) if improved else "not_saved",
            now.replace(tzinfo=None), int(len(X_train)),
            float(new_pf), float(baseline_pf), improved, improved,
        ]

        if improved:
            joblib.dump(calibrated, new_model_path)
            print(f"[RETRAIN] Saved: {new_model_path.name}")
            con.execute("UPDATE model_registry SET active = FALSE WHERE active = TRUE")
            con.execute("INSERT INTO model_registry VALUES (?,?,?,?,?,?,?,?)", insert_values)
            context["model_path"] = str(new_model_path)
            context["baseline_pf"] = new_pf
            action = "model_swapped"
        else:
            print(f"[RETRAIN] Rejected — PF {new_pf} < {round(baseline_pf * 0.95, 4)}. Keeping current model.")
            con.execute("INSERT INTO model_registry VALUES (?,?,?,?,?,?,?,?)", insert_values)
            action = "model_rejected"

        con.close()
        return {
            "status":         "success",
            "action":         action,
            "new_model_path": str(new_model_path) if improved else "not_saved",
            "new_pf":         new_pf,
            "baseline_pf":    baseline_pf,
            "improved":       improved,
            "rows_trained":   int(len(X_train)),
            "rows_oos":       int(len(X_oos)),
            "model_id":       model_id,
        }

    except Exception:
        con.close()
        return {"status": "failed", "error": traceback.format_exc()}