# =============================================================================
# paper_trade_agent.py — M4 Paper Trading Agent (Enhanced SL/TP v2)
# Enhancements over v1:
#   1. Break-even stop: once price hits 0.5x TP, stop moves to fill_entry
#   2. Partial TP: 50% exit at 1x RR, remainder runs to 2x RR
#   3. Hard stop cap: never wider than max_stop_pct (default 2.5%)
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


def _simulate_trade(entry_price, future_prices, future_times, stop_pct, tp_pct,
                    fee, slip, max_hold_bars, breakeven_trigger=0.5):
    """
    Simulates a single long trade with:
    - Break-even stop: activates when price reaches (entry + 0.5 * tp_distance)
    - Partial TP: 50% exit at 1x RR, rest at 2x RR
    - Hard stop cap enforced before calling this function
    """
    fill_entry = entry_price * (1 + slip)
    stop_price = entry_price * (1 - stop_pct)
    tp1_price  = entry_price * (1 + stop_pct * 1.0)   # 1x RR
    tp2_price  = entry_price * (1 + tp_pct)            # 2x RR (full TP)
    breakeven_trigger_price = entry_price * (1 + stop_pct * breakeven_trigger)

    breakeven_activated = False
    partial_tp_hit      = False
    partial_tp_price    = None
    partial_tp_time     = None
    partial_tp_idx      = None

    exit_price  = float(future_prices[min(max_hold_bars, len(future_prices) - 1)])
    exit_time   = future_times[min(max_hold_bars, len(future_times) - 1)]
    exit_reason = "timeexit"
    chosen_idx  = min(max_hold_bars, len(future_prices) - 1)

    for j in range(1, min(max_hold_bars, len(future_prices) - 1) + 1):
        px = float(future_prices[j])

        # Activate break-even stop
        if not breakeven_activated and px >= breakeven_trigger_price:
            stop_price = fill_entry   # move stop to entry — zero loss
            breakeven_activated = True

        # Partial TP at 1x RR (first half exits)
        if not partial_tp_hit and px >= tp1_price:
            partial_tp_hit   = True
            partial_tp_price = px * (1 - slip)
            partial_tp_time  = future_times[j]
            partial_tp_idx   = j
            # Don't break — let remainder run toward tp2

        # Full TP (2x RR)
        if px >= tp2_price:
            exit_price  = px * (1 - slip)
            exit_time   = future_times[j]
            exit_reason = "takeprofit"
            chosen_idx  = j
            break

        # Stop loss
        if px <= stop_price:
            exit_price  = stop_price * (1 - slip)
            exit_time   = future_times[j]
            exit_reason = "stoploss_be" if breakeven_activated else "stoploss"
            chosen_idx  = j
            break

    fill_exit = float(exit_price)

    # Blended return: 50% at partial TP price (if hit), 50% at final exit
    if partial_tp_hit and partial_tp_idx is not None and partial_tp_idx < chosen_idx:
        gross_ret = 0.5 * ((partial_tp_price - fill_entry) / fill_entry) + \
                    0.5 * ((fill_exit - fill_entry) / fill_entry)
    else:
        gross_ret = (fill_exit - fill_entry) / fill_entry

    net_ret = gross_ret - 2 * fee

    return dict(
        fill_entry=fill_entry,
        fill_exit=fill_exit,
        stop_price=stop_price,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
        gross_ret=gross_ret,
        net_ret=net_ret,
        exit_reason=exit_reason,
        exit_time=exit_time,
        bars_held=int(chosen_idx),
        breakeven_activated=breakeven_activated,
        partial_tp_hit=partial_tp_hit,
    )


def run_paper_trade_agent(cfg: dict, context: dict) -> dict:
    """
    Typed tool contract
    INPUT : cfg  — full agent_config.yaml as dict
    INPUT : context — shared orchestrator context dict
    OUTPUT: dict with status, trades_executed, win_rate, equity, error
    """
    ptcfg        = cfg["paper_trade_agent"]
    paths        = cfg["paths"]
    tables       = cfg["tables"]
    db_path      = PROJECT_ROOT / paths["db_path"]

    band_min     = ptcfg["band_min"]
    fee          = ptcfg["fee_bps"]    / 10_000.0
    slip         = ptcfg["slippage_bps"] / 10_000.0
    initial_eq   = ptcfg["initial_equity"]
    risk_pct     = ptcfg["risk_pct"]
    stop_vol_mult = ptcfg["stop_vol_mult"]
    rr_multiple  = ptcfg["rr_multiple"]         # should be 2.0
    max_hold     = ptcfg["max_hold_bars"]
    max_stop_pct = ptcfg.get("max_stop_pct", 0.025)   # NEW: hard cap 2.5%
    be_trigger   = ptcfg.get("breakeven_trigger", 0.5) # NEW: 0.5x RR triggers BE
    cooldown_bars = ptcfg.get("cooldown_bars", 3)

    con = duckdb.connect(str(db_path))
    try:
        df = con.execute(
            f"SELECT * FROM {tables['predictions']} ORDER BY open_time"
        ).fetchdf()

        if df.empty:
            con.close()
            return {"status": "skipped", "reason": "no predictions", "trades_executed": 0}

        account_equity = initial_eq
        trades = []
        n = len(df)
        i = 0
        last_exit_idx = -999   # tracks when last trade exited

        while i < n:
            row = df.iloc[i]

            if str(row.get("regime_label", "")) != "compression":
                i += 1
                continue
            if (i - last_exit_idx) < cooldown_bars:
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
            vol         = float(row["vol_20"])
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.01

            # Stop pct: vol-scaled but capped at max_stop_pct
            stop_pct = min(stop_vol_mult * vol, max_stop_pct)   # ENHANCED
            tp_pct   = stop_pct * rr_multiple

            future_end    = min(i + max_hold, n - 1)
            future_prices = df.iloc[i+1:future_end+1]["close"].astype(float).tolist()
            future_times  = df.iloc[i+1:future_end+1]["open_time"].tolist()

            if len(future_prices) < 2:
                i += 1
                continue

            risk_dollars       = account_equity * risk_pct
            stop_dist_dollars  = entry_price * stop_pct
            if stop_dist_dollars <= 0:
                i += 1
                continue

            qty = risk_dollars / stop_dist_dollars

            sim = _simulate_trade(
                entry_price=entry_price,
                future_prices=future_prices,
                future_times=future_times,
                stop_pct=stop_pct,
                tp_pct=tp_pct,
                fee=fee,
                slip=slip,
                max_hold_bars=max_hold,
                breakeven_trigger=be_trigger,
            )

            pnl_dollars    = qty * entry_price * sim["net_ret"]
            account_equity = account_equity + pnl_dollars
            last_exit_idx = i + sim["bars_held"]   # cooldown starts from exit bar
            
            trades.append({
                "run_id":               context.get("run_id"),
                "entry_time":           row["open_time"],
                "exit_time":            sim["exit_time"],
                "regime_label":         "compression",
                "entry_price":          round(entry_price, 4),
                "exit_price":           round(sim["fill_exit"], 4),
                "specialist_pred_proba":round(pred_proba, 6),
                "specialist_pred_label":pred_label,
                "vol_20":               round(vol, 6),
                "stop_pct":             round(stop_pct, 6),
                "take_profit_pct":      round(tp_pct, 6),
                "exit_reason":          sim["exit_reason"],
                "bars_held":            sim["bars_held"],
                "breakeven_activated":  sim["breakeven_activated"],
                "partial_tp_hit":       sim["partial_tp_hit"],
                "gross_return":         round(sim["gross_ret"], 6),
                "net_return":           round(sim["net_ret"], 6),
                "pnl_dollars":          round(pnl_dollars, 4),
                "account_equity":       round(account_equity, 4),
                "trade_time_utc":       datetime.now(timezone.utc).isoformat(),
            })

            i += sim["bars_held"] + 1   # advance past this trade's hold period

        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            con.execute("""
                CREATE TABLE IF NOT EXISTS paper_trade_agent_log (
                    run_id VARCHAR, entry_time TIMESTAMP, exit_time TIMESTAMP,
                    regime_label VARCHAR, entry_price DOUBLE, exit_price DOUBLE,
                    specialist_pred_proba DOUBLE, specialist_pred_label INTEGER,
                    vol_20 DOUBLE, stop_pct DOUBLE, take_profit_pct DOUBLE,
                    exit_reason VARCHAR, bars_held INTEGER,
                    breakeven_activated BOOLEAN, partial_tp_hit BOOLEAN,
                    gross_return DOUBLE, net_return DOUBLE,
                    pnl_dollars DOUBLE, account_equity DOUBLE, trade_time_utc VARCHAR
                )
            """)
            con.register("trades_df", trades_df)
            con.execute("INSERT INTO paper_trade_agent_log SELECT * FROM trades_df")

            win_rate    = float((trades_df["net_return"] > 0).mean())
            total_return = float(trades_df["account_equity"].iloc[-1] / initial_eq - 1.0)
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
            "status":          "failed",
            "error":           traceback.format_exc(),
            "trades_executed": 0,
            "win_rate":        0.0,
            "equity":          ptcfg["initial_equity"],
        }