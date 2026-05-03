# =============================================================================
# feature_agent.py
# Computes the exact 23 features the model was trained on and writes to
# ethusd_features_1h_v2. Column names match feature_cols.json precisely.
# =============================================================================

import sys
import traceback
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 23 model features + regime_label from raw OHLCV.
    Column names must exactly match feature_cols.json.
    """
    df = df.sort_values("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"])

    # ── FIX 1: Fallbacks for optional Binance columns ─────────────────────────
    # Historical rows in ethusd_1h_market predate the schema migration.
    # number_of_trades: use volume as a monotonically correlated proxy.
    # taker_buy_base_volume: neutral 50/50 split assumption.
    if "number_of_trades" not in df.columns or df["number_of_trades"].isna().all():
        print("[FEAT] WARN: number_of_trades unavailable — using volume proxy")
        df["number_of_trades"] = df["volume"].astype("float64")
    else:
        df["number_of_trades"] = (
            df["number_of_trades"]
            .astype("float64")          # Int64 → float64, NaN preserved
            .fillna(df["volume"].astype("float64"))
        )

    if "taker_buy_base_volume" not in df.columns or df["taker_buy_base_volume"].isna().all():
        print("[FEAT] WARN: taker_buy_base_volume unavailable — using volume * 0.5")
        df["taker_buy_base_volume"] = (df["volume"] * 0.5).astype("float64")
    else:
        df["taker_buy_base_volume"] = (
            df["taker_buy_base_volume"]
            .astype("float64")
            .fillna((df["volume"] * 0.5).astype("float64"))
        )

    # ── 1. Return features ────────────────────────────────────────────────────
    df["return_1h"]     = df["close"].pct_change(1)
    df["log_return_1h"] = np.log(df["close"] / df["close"].shift(1))
    df["return_4h"]     = df["close"].pct_change(4)
    df["return_24h"]    = df["close"].pct_change(24)

    # ── 2. Volume / trade change features ────────────────────────────────────
    df["volume_change_1h"] = df["volume"].pct_change(1)
    df["trades_change_1h"] = df["number_of_trades"].pct_change(1)

    # ── 3. Price range ────────────────────────────────────────────────────────
    df["high_low_range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # ── 4. Volatility (rolling std of log returns) ────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["vol_24h"] = log_ret.rolling(24).std()
    df["vol_72h"] = log_ret.rolling(72).std()

    # ── 5. ATR (14-period, normalised) ───────────────────────────────────────
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["atr_14_pct"] = atr14 / df["close"].replace(0, np.nan)

    # ── 6. RSI (14) ──────────────────────────────────────────────────────────
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── 7. SMA cross features ────────────────────────────────────────────────
    sma7  = df["close"].rolling(7).mean()
    sma24 = df["close"].rolling(24).mean()
    sma72 = df["close"].rolling(72).mean()

    df["close_vs_sma_7"]  = (df["close"] - sma7)  / sma7.replace(0, np.nan)
    df["close_vs_sma_24"] = (df["close"] - sma24) / sma24.replace(0, np.nan)
    df["sma_7_vs_24"]     = (sma7  - sma24) / sma24.replace(0, np.nan)
    df["sma_24_vs_72"]    = (sma24 - sma72) / sma72.replace(0, np.nan)

    # ── 8. Relative volume / trades ──────────────────────────────────────────
    vol_ma24    = df["volume"].rolling(24).mean()
    trades_ma24 = df["number_of_trades"].rolling(24).mean()
    df["rel_volume_24"] = df["volume"]           / vol_ma24.replace(0, np.nan)
    df["rel_trades_24"] = df["number_of_trades"] / trades_ma24.replace(0, np.nan)

    # ── 9. Taker buy ratio ───────────────────────────────────────────────────
    df["taker_buy_ratio"] = (
        df["taker_buy_base_volume"] / df["volume"].replace(0, np.nan)
    )

    # ── 10. Time features ────────────────────────────────────────────────────
    df["hour_of_day"] = df["open_time"].dt.hour
    df["day_of_week"] = df["open_time"].dt.dayofweek

    # ── 11. Bollinger Band distance features ─────────────────────────────────
    bb_mid   = df["close"].rolling(24).mean()
    bb_std   = df["close"].rolling(24).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["dist_to_bb_upper_pct"] = (bb_upper - df["close"]) / df["close"].replace(0, np.nan)
    df["dist_to_bb_lower_pct"] = (df["close"] - bb_lower) / df["close"].replace(0, np.nan)

    # ── 12. Regime proxy (vol compression percentile rank) ───────────────────
    df["regime_proxy_24h"] = (
        df["vol_24h"]
        .rolling(72, min_periods=24)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )

    # ── 13. regime_label ─────────────────────────────────────────────────────
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    bb_pct   = bb_width.rolling(72, min_periods=24).apply(
                   lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    vol_pct  = df["vol_24h"].rolling(72, min_periods=24).apply(
                   lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    df["regime_label"] = np.where(
        (bb_pct < 0.35) & (vol_pct < 0.35), "compression", "trending"
    )

    return df


def run_feature_agent(cfg: dict, context: dict) -> dict:
    paths  = cfg["paths"]
    tables = cfg["tables"]

    db_path      = PROJECT_ROOT / paths["db_path"]
    market_table = tables.get("canonical_market", "ethusd_1h_market")
    v2_table     = "ethusd_features_1h_v2"

    MODEL_FEATURE_COLS = [
        "return_1h", "log_return_1h", "return_4h", "return_24h",
        "volume_change_1h", "trades_change_1h", "high_low_range_pct",
        "vol_24h", "vol_72h", "atr_14_pct", "rsi_14",
        "close_vs_sma_7", "close_vs_sma_24", "sma_7_vs_24", "sma_24_vs_72",
        "rel_volume_24", "rel_trades_24", "taker_buy_ratio",
        "hour_of_day", "day_of_week",
        "dist_to_bb_upper_pct", "dist_to_bb_lower_pct",
        "regime_proxy_24h",
    ]
    WRITE_COLS = (
        ["open_time", "open", "high", "low", "close", "volume", "regime_label"]
        + MODEL_FEATURE_COLS
    )

    con = duckdb.connect(str(db_path))

    try:
        df = con.execute(
            f"SELECT * FROM {market_table} ORDER BY open_time"
        ).df()

        if df.empty:
            con.close()
            return {"status": "skipped", "reason": "market_table_empty", "rows_updated": 0}

        # ── FIX 2: soft check — only fail on truly critical OHLCV columns ────
        # number_of_trades and taker_buy_base_volume handled inside
        # _compute_features() with fallbacks — never hard-fail on them.
        required_raw = {"open_time", "open", "high", "low", "close", "volume"}
        missing_required = required_raw - set(df.columns)
        if missing_required:
            con.close()
            return {
                "status":       "failed",
                "error":        f"market table missing critical columns: {missing_required}",
                "rows_updated": 0,
            }

        df = _compute_features(df)

        # ── Incremental check against v2_table ───────────────────────────────
        try:
            existing    = con.execute(f"SELECT open_time FROM {v2_table}").df()
            existing_ts = set(pd.to_datetime(existing["open_time"]).dt.floor("s"))
        except Exception:
            existing_ts = set()

        df["open_time"] = pd.to_datetime(df["open_time"]).dt.floor("s")
        cols_to_write   = [c for c in WRITE_COLS if c in df.columns]
        features_df     = df[cols_to_write].copy()
        new_rows        = features_df[~features_df["open_time"].isin(existing_ts)].copy()

        if new_rows.empty:
            con.close()
            return {
                "status":       "success",
                "rows_updated": 0,
                "message":      "Feature store already up to date",
            }

        new_rows = new_rows.dropna(subset=MODEL_FEATURE_COLS, how="all").copy()

        # ── Write to v2_table ─────────────────────────────────────────────────
        try:
            existing_cols = {
                r[0] for r in con.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{v2_table}'"
                ).fetchall()
            }
            if existing_cols != set(cols_to_write):
                print(f"[FEAT] Schema mismatch — recreating {v2_table}")
                con.execute(f"DROP TABLE IF EXISTS {v2_table}")
                con.register("feat_reg", new_rows)
                con.execute(f"CREATE TABLE {v2_table} AS SELECT * FROM feat_reg")
            else:
                con.register("feat_reg", new_rows)
                con.execute(f"INSERT INTO {v2_table} SELECT * FROM feat_reg")
        except Exception:
            con.execute(f"DROP TABLE IF EXISTS {v2_table}")
            con.register("feat_reg", new_rows)
            con.execute(f"CREATE TABLE {v2_table} AS SELECT * FROM feat_reg")

        rows_updated  = len(new_rows)
        regime_counts = new_rows["regime_label"].value_counts().to_dict()
        print(f"[FEAT] Inserted {rows_updated} rows into {v2_table} | regimes: {regime_counts}")

        con.close()
        return {
            "status":        "success",
            "rows_updated":  rows_updated,
            "message":       f"Added {rows_updated} rows to {v2_table}",
            "regime_counts": regime_counts,
        }

    except Exception:
        con.close()
        return {
            "status":       "failed",
            "error":        traceback.format_exc(),
            "rows_updated": 0,
        }