import sys
import traceback
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def run_feature_agent(cfg: dict, context: dict) -> dict:
    paths  = cfg["paths"]
    tables = cfg["tables"]

    db_path       = PROJECT_ROOT / paths["db_path"]
    market_table  = tables.get("market", "ethusd_1h_market")
    feature_table = tables.get("feature_store", "feature_store_v2")

    con = duckdb.connect(str(db_path))

    try:
        df = con.execute(
            f"SELECT * FROM {market_table} ORDER BY open_time"
        ).df()

        if df.empty:
            con.close()
            return {"status": "skipped", "reason": "market_table_empty", "rows_updated": 0}

        df = df.sort_values("open_time").reset_index(drop=True)

        df["ret_1"]  = df["close"].pct_change(1)
        df["ret_3"]  = df["close"].pct_change(3)
        df["ret_6"]  = df["close"].pct_change(6)
        df["ret_12"] = df["close"].pct_change(12)
        df["ret_24"] = df["close"].pct_change(24)

        df["vol5"]  = df["close"].rolling(5).std()
        df["vol10"] = df["close"].rolling(10).std()
        df["vol20"] = df["close"].rolling(20).std()
        df["vol50"] = df["close"].rolling(50).std()

        df["vol_ratio_5_20"]  = df["vol5"]  / df["vol20"].replace(0, np.nan)
        df["vol_ratio_10_50"] = df["vol10"] / df["vol50"].replace(0, np.nan)

        df["hl_range"]      = df["high"] - df["low"]
        df["hl_range_ma10"] = df["hl_range"].rolling(10).mean()
        df["hl_range_norm"] = df["hl_range"] / df["close"].replace(0, np.nan)

        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        ma5  = df["close"].rolling(5).mean()
        ma10 = df["close"].rolling(10).mean()
        ma20 = df["close"].rolling(20).mean()
        ma50 = df["close"].rolling(50).mean()

        df["ma5_slope"]     = ma5.diff()
        df["ma10_slope"]    = ma10.diff()
        df["price_vs_ma20"] = (df["close"] - ma20) / ma20.replace(0, np.nan)
        df["price_vs_ma50"] = (df["close"] - ma50) / ma50.replace(0, np.nan)

        vol_ma24 = df["volume"].rolling(24).mean()
        df["vol_ratio"] = df["volume"] / vol_ma24.replace(0, np.nan)
        df["vol_trend"] = df["volume"].pct_change(6)

        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
        df["atr_norm"] = df["atr14"] / df["close"].replace(0, np.nan)

        bb_pct  = df["bb_width"].rank(pct=True)
        vol_pct = df["vol20"].rank(pct=True)
        df["regime_label"] = np.where(
            (bb_pct < 0.35) & (vol_pct < 0.35), "compression", "trending"
        )

        FEATURE_COLS = [
            "open_time", "open", "high", "low", "close", "volume",
            "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
            "vol5", "vol10", "vol20", "vol50",
            "vol_ratio_5_20", "vol_ratio_10_50",
            "hl_range", "hl_range_ma10", "hl_range_norm",
            "bb_width", "bb_upper", "bb_lower",
            "rsi", "ma5_slope", "ma10_slope",
            "price_vs_ma20", "price_vs_ma50",
            "vol_ratio", "vol_trend",
            "atr14", "atr_norm",
            "regime_label",
        ]

        cols_to_use = [c for c in FEATURE_COLS if c in df.columns]
        features_df = df[cols_to_use].copy()

        try:
            existing = con.execute(
                f"SELECT open_time FROM {feature_table}"
            ).df()
            existing_ts = set(pd.to_datetime(existing["open_time"]).dt.floor("s"))
        except Exception:
            existing_ts = set()

        features_df["open_time"] = pd.to_datetime(features_df["open_time"]).dt.floor("s")
        new_rows = features_df[~features_df["open_time"].isin(existing_ts)].copy()

        if new_rows.empty:
            con.close()
            return {
                "status": "success",
                "rows_updated": 0,
                "message": "Feature store already up to date"
            }

        new_rows = new_rows.dropna(how="all").copy()

        try:
            existing_cols = [
                r[0] for r in con.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{feature_table}' ORDER BY ordinal_position"
                ).fetchall()
            ]
            if set(existing_cols) != set(cols_to_use):
                print(f"[FEAT] Schema mismatch detected — recreating {feature_table}")
                con.execute(f"DROP TABLE IF EXISTS {feature_table}")
                con.register("new_rows_reg", new_rows)
                con.execute(
                    f"CREATE TABLE {feature_table} AS SELECT * FROM new_rows_reg"
                )
            else:
                con.register("new_rows_reg", new_rows)
                con.execute(
                    f"INSERT INTO {feature_table} "
                    f"SELECT * FROM new_rows_reg"
                )
        except Exception:
            con.execute(f"DROP TABLE IF EXISTS {feature_table}")
            con.register("new_rows_reg", new_rows)
            con.execute(
                f"CREATE TABLE {feature_table} AS SELECT * FROM new_rows_reg"
            )

        rows_updated = len(new_rows)
        print(f"[FEAT] Inserted {rows_updated} new rows into {feature_table}")

        con.close()
        return {
            "status": "success",
            "rows_updated": rows_updated,
            "message": f"Added {rows_updated} rows",
        }

    except Exception:
        con.close()
        return {
            "status": "failed",
            "error": traceback.format_exc(),
            "rows_updated": 0,
        }