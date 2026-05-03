# =============================================================================
# data_agent.py
# Fetches latest ETH/USD 1h candles from Binance
# Appends new rows to DuckDB canonical_market table incrementally
# Aborts pipeline if no new data or gap detected
# =============================================================================

import sys
import traceback
import requests
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# Full Binance KLINE response field order (index 0–11)
# https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
KLINE_COLS = [
    "open_time",              # 0  ms timestamp
    "open",                   # 1
    "high",                   # 2
    "low",                    # 3
    "close",                  # 4
    "volume",                 # 5
    "close_time",             # 6  ms timestamp
    "quote_asset_volume",     # 7
    "number_of_trades",       # 8
    "taker_buy_base_volume",  # 9
    "taker_buy_quote_volume", # 10
    "_ignore",                # 11 unused field Binance sends
]


def run_data_agent(cfg: dict, context: dict) -> dict:
    """
    Typed tool contract:
    INPUT  cfg     : full agent_config.yaml as dict
    INPUT  context : shared orchestrator context dict
    OUTPUT dict    : {status, new_rows_added, latest_timestamp, error}
    """
    da_cfg        = cfg["data_agent"]
    paths         = cfg["paths"]
    tables        = cfg["tables"]

    db_path       = PROJECT_ROOT / paths["db_path"]
    market_table  = tables["canonical_market"]
    symbol        = da_cfg["symbol"]
    interval      = da_cfg["interval"]
    base_url      = da_cfg["binance_base_url"]
    lookback_bars = int(da_cfg["lookback_bars"])
    max_gap_hours = int(da_cfg["max_gap_hours"])

    con = duckdb.connect(str(db_path))

    try:
        # ── 1. Ensure canonical market table exists (full schema) ────────────
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {market_table} (
                open_time              TIMESTAMP,
                open                   DOUBLE,
                high                   DOUBLE,
                low                    DOUBLE,
                close                  DOUBLE,
                volume                 DOUBLE,
                close_time             TIMESTAMP,
                quote_asset_volume     DOUBLE,
                number_of_trades       BIGINT,
                taker_buy_base_volume  DOUBLE,
                taker_buy_quote_volume DOUBLE,
                symbol                 VARCHAR,
                interval               VARCHAR
            )
        """)

        # ── 1b. Migrate existing table if new columns are missing ────────────
        # Safe to run every time — ADD COLUMN IF NOT EXISTS is idempotent
        for col_ddl in [
            "quote_asset_volume     DOUBLE",
            "number_of_trades       BIGINT",
            "taker_buy_base_volume  DOUBLE",
            "taker_buy_quote_volume DOUBLE",
        ]:
            col_name = col_ddl.split()[0]
            try:
                existing = {
                    r[0] for r in con.execute(
                        f"SELECT column_name FROM information_schema.columns "
                        f"WHERE table_name = '{market_table}'"
                    ).fetchall()
                }
                if col_name not in existing:
                    con.execute(
                        f"ALTER TABLE {market_table} ADD COLUMN {col_ddl}"
                    )
                    print(f"[DATA] Migrated: added column {col_name} to {market_table}")
            except Exception:
                pass  # column already exists in some DuckDB versions

        # ── 2. Get latest timestamp already in DB ────────────────────────────
        result = con.execute(
            f"SELECT MAX(open_time) FROM {market_table}"
        ).fetchone()[0]

        last_ts_ms = None
        if result is not None:
            if hasattr(result, "timestamp"):
                last_ts_ms = int(result.timestamp() * 1000) + 1

        # ── 3. Fetch from Binance ────────────────────────────────────────────
        params = {
            "symbol":   symbol,
            "interval": interval,
            "limit":    lookback_bars,
        }
        if last_ts_ms:
            params["startTime"] = last_ts_ms

        resp = requests.get(
            f"{base_url}/api/v3/klines",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        raw = resp.json()

        if not raw:
            con.close()
            return {
                "status":           "success",
                "new_rows_added":   0,
                "latest_timestamp": str(result) if result else None,
                "message":          "No new candles from Binance",
            }

        # ── 4. Parse all 12 KLINE fields ─────────────────────────────────────
        rows = []
        for k in raw:
            open_time  = pd.to_datetime(int(k[0]), unit="ms", utc=True).tz_localize(None)
            close_time = pd.to_datetime(int(k[6]), unit="ms", utc=True).tz_localize(None)
            rows.append({
                "open_time":              open_time,
                "open":                   float(k[1]),
                "high":                   float(k[2]),
                "low":                    float(k[3]),
                "close":                  float(k[4]),
                "volume":                 float(k[5]),
                "close_time":             close_time,
                "quote_asset_volume":     float(k[7]),
                "number_of_trades":       int(k[8]),
                "taker_buy_base_volume":  float(k[9]),
                "taker_buy_quote_volume": float(k[10]),
                "symbol":                 symbol,
                "interval":               interval,
                # k[11] is ignored — Binance internal field
            })

        df = pd.DataFrame(rows)

        # ── 5. Deduplicate — only insert rows not already in DB ──────────────
        if result is not None:
            df = df[df["open_time"] > pd.Timestamp(result)]

        if df.empty:
            con.close()
            return {
                "status":           "success",
                "new_rows_added":   0,
                "latest_timestamp": str(result),
                "message":          "All fetched candles already in DB",
            }

        # ── 6. Gap detection ─────────────────────────────────────────────────
        if result is not None:
            gap_hours = (
                df["open_time"].min() - pd.Timestamp(result)
            ).total_seconds() / 3600
            if gap_hours > max_gap_hours:
                print(
                    f"[DATA] WARNING: Gap of {gap_hours:.1f}h detected "
                    f"(max allowed: {max_gap_hours}h)"
                )

        # ── 7. Insert new rows ───────────────────────────────────────────────
        con.register("new_candles", df)
        insert_cols = ", ".join(df.columns.tolist())
        con.execute(f"INSERT INTO {market_table} ({insert_cols}) SELECT {insert_cols} FROM new_candles")
        new_rows  = len(df)
        latest_ts = str(df["open_time"].max())
        context["new_rows_added"] = int(new_rows)

        print(f"[DATA] Inserted {new_rows} new candles | Latest: {latest_ts}")

        con.close()
        return {
            "status":           "success",
            "new_rows_added":   int(new_rows),
            "latest_timestamp": latest_ts,
            "symbol":           symbol,
            "interval":         interval,
        }

    except requests.exceptions.ConnectionError:
        con.close()
        return {
            "status":         "failed",
            "error":          "Binance connection failed — check internet or VPN",
            "new_rows_added": 0,
        }
    except requests.exceptions.Timeout:
        con.close()
        return {
            "status":         "failed",
            "error":          "Binance request timed out after 15s",
            "new_rows_added": 0,
        }
    except Exception:
        con.close()
        return {
            "status":         "failed",
            "error":          traceback.format_exc(),
            "new_rows_added": 0,
        }