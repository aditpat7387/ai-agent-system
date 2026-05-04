import sys
import traceback
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Trade simulator — unchanged logic, kept here for self-containment
# ---------------------------------------------------------------------------

def _simulate_trade(entry_price, future_prices, future_times,
                    stop_pct, tp_pct, fee, slip, max_hold_bars,
                    be_trigger=0.5):
    """
    Single long trade with:
      - Break-even stop at entry + be_trigger * stop_distance
      - Partial TP (50 %) at 1x RR, remainder at 2x RR
      - Hard max-hold time exit
    """
    fill_entry   = entry_price * (1 + slip)
    stop_price   = entry_price * (1 - stop_pct)
    tp1_price    = entry_price * (1 + stop_pct * 1.0)   # 1x RR
    tp2_price    = entry_price * (1 + tp_pct)            # 2x RR full TP
    be_trigger_price = entry_price * (1 + stop_pct * be_trigger)

    be_activated   = False
    partial_tp_hit = False
    partial_tp_px  = None
    partial_tp_idx = None

    chosen_idx = min(max_hold_bars, len(future_prices) - 1)
    exit_price = float(future_prices[chosen_idx])
    exit_time  = future_times[chosen_idx]
    exit_reason = "timeexit"

    for j in range(1, min(max_hold_bars, len(future_prices) - 1) + 1):
        px = float(future_prices[j])

        if not be_activated and px >= be_trigger_price:
            stop_price   = fill_entry          # move stop to entry
            be_activated = True

        if not partial_tp_hit and px >= tp1_price:
            partial_tp_hit = True
            partial_tp_px  = px * (1 - slip)
            partial_tp_idx = j

        if px >= tp2_price:
            exit_price  = px * (1 - slip)
            exit_time   = future_times[j]
            exit_reason = "takeprofit"
            chosen_idx  = j
            break

        if px <= stop_price:
            exit_price  = stop_price * (1 - slip)
            exit_time   = future_times[j]
            exit_reason = "stoploss_be" if be_activated else "stoploss"
            chosen_idx  = j
            break

    fill_exit = float(exit_price)

    if partial_tp_hit and partial_tp_idx is not None and partial_tp_idx < chosen_idx:
        gross_ret = (0.5 * (partial_tp_px - fill_entry) / fill_entry
                   + 0.5 * (fill_exit    - fill_entry) / fill_entry)
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
        be_activated=be_activated,
        partial_tp_hit=partial_tp_hit,
    )


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def run_paper_trade_agent(cfg: dict, context: dict) -> dict:
    """
    Incremental paper-trade agent.

    Key fix vs prior version
    ========================
    Previously the agent re-simulated ALL rows in the predictions table on
    every run, which caused:
      - Duplicate trade rows written on every pipeline cycle
      - False "no new trades" result in live mode because rel_score on
        historical rows is almost always below band_min in a fresh batch

    New behaviour
    =============
    1. Read MAX(entry_time) already stored in paper_trade_agent_log.
    2. Only process prediction rows NEWER than that timestamp.
    3. If there are no new rows, return trades_executed=0 cleanly.
    4. Account equity is restored from the last stored row so the running
       balance is consistent across incremental runs.
    """
    pt_cfg     = cfg["paper_trade_agent"]
    paths      = cfg["paths"]
    tables     = cfg["tables"]
    db_path    = PROJECT_ROOT / paths["db_path"]

    band_min      = float(pt_cfg["band_min"])
    fee           = float(pt_cfg["fee_bps"]) / 10_000.0
    slip          = float(pt_cfg["slippage_bps"]) / 10_000.0
    initial_eq    = float(pt_cfg["initial_equity"])
    risk_pct      = float(pt_cfg["risk_pct"])
    stop_vol_mult = float(pt_cfg["stop_vol_mult"])
    rr_multiple   = float(pt_cfg["rr_multiple"])
    max_hold      = int(pt_cfg["max_hold_bars"])
    max_stop_pct  = float(pt_cfg.get("max_stop_pct", 0.025))
    be_trigger    = float(pt_cfg.get("breakeven_trigger", 0.5))
    cooldown_bars = int(pt_cfg.get("cooldown_bars", 3))

    pred_table  = tables["predictions"]
    trade_table = tables["paper_trades"]

    con = duckdb.connect(str(db_path))
    try:
        # ------------------------------------------------------------------
        # 1. Ensure trade table exists
        # ------------------------------------------------------------------
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {trade_table} (
                run_id                VARCHAR,
                entry_time            TIMESTAMP,
                exit_time             TIMESTAMP,
                regime_label          VARCHAR,
                entry_price           DOUBLE,
                exit_price            DOUBLE,
                specialist_pred_proba DOUBLE,
                specialist_pred_label INTEGER,
                vol20                 DOUBLE,
                stop_pct              DOUBLE,
                take_profit_pct       DOUBLE,
                exit_reason           VARCHAR,
                bars_held             INTEGER,
                breakeven_activated   BOOLEAN,
                partial_tp_hit        BOOLEAN,
                gross_return          DOUBLE,
                net_return            DOUBLE,
                pnl_dollars           DOUBLE,
                account_equity        DOUBLE,
                trade_time_utc        VARCHAR
            )
        """)

        # ------------------------------------------------------------------
        # 2. Fetch last processed entry_time + restore running equity
        # ------------------------------------------------------------------
        last_row = con.execute(
            f"SELECT MAX(entry_time) AS last_ts FROM {trade_table}"
        ).fetchone()
        last_processed_ts = last_row[0] if last_row and last_row[0] else None

        equity_row = con.execute(
            f"SELECT account_equity FROM {trade_table} "
            f"ORDER BY entry_time DESC LIMIT 1"
        ).fetchone()
        account_equity = float(equity_row[0]) if equity_row and equity_row[0] else initial_eq

        # ------------------------------------------------------------------
        # 3. Fetch ONLY new predictions since last processed timestamp
        # ------------------------------------------------------------------
        if last_processed_ts is not None:
            df = con.execute(
                f"SELECT * FROM {pred_table} "
                f"WHERE open_time > ? ORDER BY open_time",
                [last_processed_ts]
            ).fetchdf()
            print(f"[TRADE] Incremental mode — rows after {last_processed_ts}: {len(df)}")
        else:
            df = con.execute(
                f"SELECT * FROM {pred_table} ORDER BY open_time"
            ).fetchdf()
            print(f"[TRADE] First run — processing all {len(df)} prediction rows")

        if df.empty:
            con.close()
            return dict(
                status="success",
                trades_executed=0,
                win_rate=0.0,
                total_return=0.0,
                equity=round(account_equity, 2),
                band_min=band_min,
                trade_table=trade_table,
                message="No new prediction rows since last run",
            )

        # ------------------------------------------------------------------
        # 4. Map columns — handle old and new schema gracefully
        # ------------------------------------------------------------------
        col_map = {
            "open_time":  ["open_time", "opentime"],
            "close":      ["close"],
            "pred_label": ["pred_label", "predlabel", "specialist_pred_label"],
            "pred_proba": ["pred_proba", "predproba", "specialist_pred_proba"],
            "cal_proba":  ["cal_proba",  "calproba"],
            "rel_score":  ["rel_score",  "relscore"],
            "vol20":      ["vol20", "vol_20", "atr_14_pct"],
            "regime":     ["regime", "regime_label"],
        }
        canon = {}
        for target, candidates in col_map.items():
            for c in candidates:
                if c in df.columns:
                    canon[target] = c
                    break

        def gcol(row, key, default=np.nan):
            col = canon.get(key)
            return row[col] if col and col in row.index else default

        # ------------------------------------------------------------------
        # 5. Simulation loop — only on new rows
        # ------------------------------------------------------------------
        n             = len(df)
        i             = 0
        last_exit_idx = -999
        trades        = []

        while i < n:
            row = df.iloc[i]

            # regime guard — only compression
            regime = str(gcol(row, "regime", "")).lower()
            if regime not in ("compression", ""):
                i += 1
                continue

            # cooldown guard
            if i - last_exit_idx <= cooldown_bars:
                i += 1
                continue

            pred_proba = float(gcol(row, "pred_proba", np.nan))
            rel_score  = float(gcol(row, "rel_score",  np.nan))
            pred_label = gcol(row, "pred_label", np.nan)

            if pd.isna(pred_proba) or pd.isna(pred_label):
                i += 1
                continue
            pred_label = int(pred_label)

            # signal gate — prefer rel_score, fallback to pred_proba
            score = rel_score if not np.isnan(rel_score) else pred_proba
            if score < band_min or pred_label != 1:
                i += 1
                continue

            entry_price = float(gcol(row, "close", np.nan))
            if np.isnan(entry_price) or entry_price <= 0:
                i += 1
                continue

            vol = float(gcol(row, "vol20", 0.01))
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.01

            stop_pct = min(stop_vol_mult * vol, max_stop_pct)
            tp_pct   = stop_pct * rr_multiple

            future_end    = min(i + max_hold, n - 1)
            open_time_col = canon.get("open_time", "open_time")
            future_prices = df.iloc[i + 1:future_end + 1]["close"].astype(float).tolist()
            future_times  = df.iloc[i + 1:future_end + 1][open_time_col].tolist()

            if len(future_prices) < 2:
                i += 1
                continue

            risk_dollars  = account_equity * risk_pct
            stop_dist_dol = entry_price * stop_pct
            if stop_dist_dol <= 0:
                i += 1
                continue
            qty = risk_dollars / stop_dist_dol

            sim = _simulate_trade(
                entry_price=entry_price,
                future_prices=future_prices,
                future_times=future_times,
                stop_pct=stop_pct,
                tp_pct=tp_pct,
                fee=fee,
                slip=slip,
                max_hold_bars=max_hold,
                be_trigger=be_trigger,
            )

            pnl_dollars    = qty * entry_price * sim["net_ret"]
            account_equity = account_equity + pnl_dollars
            last_exit_idx  = i + sim["bars_held"]

            trades.append(dict(
                run_id                = context.get("run_id", ""),
                entry_time            = row[open_time_col],
                exit_time             = sim["exit_time"],
                regime_label          = "compression",
                entry_price           = round(entry_price, 4),
                exit_price            = round(sim["fill_exit"], 4),
                specialist_pred_proba = round(pred_proba, 6),
                specialist_pred_label = pred_label,
                vol20                 = round(vol, 6),
                stop_pct              = round(stop_pct, 6),
                take_profit_pct       = round(tp_pct, 6),
                exit_reason           = sim["exit_reason"],
                bars_held             = sim["bars_held"],
                breakeven_activated   = sim["be_activated"],
                partial_tp_hit        = sim["partial_tp_hit"],
                gross_return          = round(sim["gross_ret"], 6),
                net_return            = round(sim["net_ret"], 6),
                pnl_dollars           = round(pnl_dollars, 4),
                account_equity        = round(account_equity, 4),
                trade_time_utc        = datetime.now(timezone.utc).isoformat(),
            ))

            i = last_exit_idx + 1   # jump past hold + cooldown

        # ------------------------------------------------------------------
        # 6. Persist new trades only
        # ------------------------------------------------------------------
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            con.register("new_trades_df", trades_df)
            con.execute(f"INSERT INTO {trade_table} SELECT * FROM new_trades_df")
            win_rate     = float((trades_df["net_return"] > 0).mean())
            total_return = float(trades_df["account_equity"].iloc[-1] / initial_eq - 1.0)
            print(f"[TRADE] Executed {len(trades)} new trades | "
                  f"WR={win_rate:.2%} | Equity={account_equity:.2f}")
        else:
            win_rate     = 0.0
            total_return = 0.0
            print(f"[TRADE] No qualifying signals in {len(df)} new rows "
                  f"(band_min={band_min}, regime=compression required)")

        con.close()
        return dict(
            status          = "success",
            trades_executed = len(trades),
            win_rate        = round(win_rate, 4),
            total_return    = round(total_return, 6),
            equity          = round(account_equity, 2),
            band_min        = band_min,
            trade_table     = trade_table,
        )

    except Exception:
        con.close()
        return dict(
            status          = "failed",
            error           = traceback.format_exc(),
            trades_executed = 0,
            win_rate        = 0.0,
            equity          = initial_eq,
        )