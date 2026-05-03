# =============================================================================
# paper_trade_agent.py
# Wraps run_paper_trading_shadow_v1.py as a typed agent tool
# Called by orchestrator after prediction_agent completes
# Reads predictions from DuckDB, writes trades to DuckDB
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


def run_paper_trade_agent(cfg: dict, context: dict) -> dict:
    """
    Typed tool contract:
    INPUT  cfg     : full agent_config.yaml as dict
    INPUT  context : shared orchestrator context dict
    OUTPUT dict    : {status, trades_executed, win_rate, equity, error}
    """
    pt_cfg = cfg["paper_trade_agent"]
    paths  = cfg["paths"]
    tables = cfg["tables"]

    db_path = PROJECT_ROOT / paths["db_path"]
    con = duckdb.connect(str(db_path))

    try:
        # ── 1. Load latest predictions from DuckDB ──────────────────────────
        pred_table = tables["predictions"]
        row_count = con.execute(
            f"SELECT COUNT(*) FROM {pred_table}"
        ).fetchone()[0]

        if row_count == 0:
            return {
                "status": "skipped",
                "reason": "no_predictions_available",
                "trades_executed": 0,
                "win_rate": 0.0,
                "equity": pt_cfg["initial_equity"],
            }

        df = con.execute(f"""
            SELECT *
            FROM {pred_table}
            ORDER BY open_time
        """).df()

        # ── 2. Normalise prediction columns ─────────────────────────────────
        if "pred_proba" in df.columns and "specialist_pred_proba" not in df.columns:
            df["specialist_pred_proba"] = df["pred_proba"]
        if "pred_label" in df.columns and "specialist_pred_label" not in df.columns:
            df["specialist_pred_label"] = df["pred_label"]
        if "event_label" in df.columns and "actual_label" not in df.columns:
            df["actual_label"] = df["event_label"]

        df["specialist_pred_proba"] = pd.to_numeric(
            df["specialist_pred_proba"], errors="coerce"
        )
        df["specialist_pred_label"] = pd.to_numeric(
            df["specialist_pred_label"], errors="coerce"
        )
        df["open_time"] = pd.to_datetime(df["open_time"])

        # ── 3. Build vol20 if missing ────────────────────────────────────────
        if "vol20" not in df.columns or df["vol20"].isna().all():
            ret = df["close"].astype(float).pct_change()
            df["vol20"] = ret.rolling(20).std().fillna(ret.std())

        # ── 4. Run paper trading simulation ─────────────────────────────────
        band_min        = pt_cfg["band_min"]
        fee_bps         = pt_cfg["fee_bps"]
        slippage_bps    = pt_cfg["slippage_bps"]
        initial_equity  = pt_cfg["initial_equity"]
        risk_pct        = pt_cfg["risk_pct"]
        stop_vol_mult   = pt_cfg["stop_vol_mult"]
        rr_multiple     = pt_cfg["rr_multiple"]
        max_hold_bars   = pt_cfg["max_hold_bars"]
        fee             = fee_bps / 10000.0
        slip            = slippage_bps / 10000.0

        account_equity = initial_equity
        trades = []
        n = len(df)
        i = 0

        while i < n:
            row = df.iloc[i]

            if str(row.get("regime_label", "")) != "compression":
                i += 1
                continue

            pred_proba = float(row.get("specialist_pred_proba", np.nan))
            pred_label = row.get("specialist_pred_label", np.nan)

            if pd.isna(pred_proba) or pd.isna(pred_label):
                i += 1
                continue

            pred_label = int(pred_label)

            if pred_proba < band_min or pred_label != 1:
                i += 1
                continue

            entry_price = float(row["close"])
            vol = float(row["vol20"])
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.01

            stop_pct = max(stop_vol_mult * vol, 0.005)
            tp_pct   = stop_pct * rr_multiple

            future_end    = min(i + max_hold_bars, n - 1)
            future_prices = df.iloc[i:future_end + 1]["close"].astype(float).tolist()
            future_times  = df.iloc[i:future_end + 1]["open_time"].tolist()

            if len(future_prices) < 2:
                i += 1
                continue

            stop_price = entry_price * (1 - stop_pct)
            tp_price   = entry_price * (1 + tp_pct)
            fill_entry = entry_price * (1 + slip)

            exit_price  = float(future_prices[min(max_hold_bars, len(future_prices) - 1)])
            exit_time   = future_times[min(max_hold_bars, len(future_times) - 1)]
            exit_reason = "time_exit"
            exit_idx    = min(max_hold_bars, len(future_prices) - 1)

            for j in range(1, min(max_hold_bars, len(future_prices) - 1) + 1):
                px = float(future_prices[j])
                if px <= stop_price:
                    exit_price  = stop_price
                    exit_time   = future_times[j]
                    exit_reason = "stop_loss"
                    exit_idx    = j
                    break
                if px >= tp_price:
                    exit_price  = tp_price
                    exit_time   = future_times[j]
                    exit_reason = "take_profit"
                    exit_idx    = j
                    break

            fill_exit  = exit_price * (1 - slip)
            gross_ret  = (fill_exit - fill_entry) / fill_entry
            net_ret    = gross_ret - (2 * fee)

            risk_dollars         = account_equity * risk_pct
            stop_distance_dollars = entry_price * stop_pct
            if stop_distance_dollars <= 0:
                i += 1
                continue

            qty        = risk_dollars / stop_distance_dollars
            pnl_dollars = qty * entry_price * net_ret
            account_equity += pnl_dollars

            trades.append({
                "run_id":               context.get("run_id"),
                "entry_time":           row["open_time"],
                "exit_time":            exit_time,
                "regime_label":         "compression",
                "entry_price":          round(entry_price, 4),
                "exit_price":           round(fill_exit, 4),
                "specialist_pred_proba":round(pred_proba, 6),
                "specialist_pred_label":pred_label,
                "vol20":                round(vol, 6),
                "stop_pct":             round(stop_pct, 6),
                "take_profit_pct":      round(tp_pct, 6),
                "exit_reason":          exit_reason,
                "bars_held":            exit_idx,
                "gross_return":         round(gross_ret, 6),
                "net_return":           round(net_ret, 6),
                "pnl_dollars":          round(pnl_dollars, 4),
                "account_equity":       round(account_equity, 4),
                "trade_time_utc":       datetime.now(timezone.utc).isoformat(),
            })

            i = i + exit_idx + 1

        # ── 5. Write trades to DuckDB ────────────────────────────────────────
        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            con.execute("""
                CREATE TABLE IF NOT EXISTS paper_trade_agent_log (
                    run_id                 VARCHAR,
                    entry_time             TIMESTAMP,
                    exit_time              TIMESTAMP,
                    regime_label           VARCHAR,
                    entry_price            DOUBLE,
                    exit_price             DOUBLE,
                    specialist_pred_proba  DOUBLE,
                    specialist_pred_label  INTEGER,
                    vol20                  DOUBLE,
                    stop_pct               DOUBLE,
                    take_profit_pct        DOUBLE,
                    exit_reason            VARCHAR,
                    bars_held              INTEGER,
                    gross_return           DOUBLE,
                    net_return             DOUBLE,
                    pnl_dollars            DOUBLE,
                    account_equity         DOUBLE,
                    trade_time_utc         VARCHAR
                )
            """)
            con.register("trades_df", trades_df)
            con.execute("INSERT INTO paper_trade_agent_log SELECT * FROM trades_df")

            win_rate = float((trades_df["net_return"] > 0).mean())
            total_return = float(
                trades_df["account_equity"].iloc[-1] / initial_equity - 1.0
            )
        else:
            win_rate     = 0.0
            total_return = 0.0

        con.close()

        return {
            "status":          "success",
            "trades_executed": len(trades),
            "win_rate":        round(win_rate, 4),
            "total_return":    round(total_return, 6),
            "equity":          round(account_equity, 2),
            "band_min":        band_min,
        }

    except Exception as e:
        con.close()
        return {
            "status": "failed",
            "error":  traceback.format_exc(),
            "trades_executed": 0,
            "win_rate":  0.0,
            "equity":    pt_cfg["initial_equity"],
        }
