# =============================================================================
# drift_checker.py
# Monitors model performance against validated walk-forward baseline
# Baseline: ge_0.78 band, PF=1.735, Win Rate=62.5%, OOS Return=+2.15%
# =============================================================================

import sys
import traceback
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.run_logger import log_drift


def run_drift_checker(cfg: dict, context: dict) -> dict:
    """
    Typed tool contract:
    INPUT  cfg     : full agent_config.yaml as dict
    INPUT  context : shared orchestrator context dict
    OUTPUT dict    : {status, drift_detected, current_pf, baseline_pf,
                      current_win_rate, baseline_win_rate, trades_evaluated,
                      action_taken}
    """
    dc_cfg  = cfg["drift_checker"]
    paths   = cfg["paths"]

    db_path = PROJECT_ROOT / paths["db_path"]
    con     = duckdb.connect(str(db_path))

    baseline_pf         = float(dc_cfg["baseline_profit_factor"])
    baseline_win_rate   = float(dc_cfg["baseline_win_rate"])
    baseline_oos_return = float(dc_cfg["baseline_oos_return"])
    eval_window         = int(dc_cfg["evaluation_window_rows"])
    min_trades          = int(dc_cfg["min_trades_for_drift_check"])
    pf_threshold        = float(dc_cfg["drift_pf_threshold"])
    wr_threshold        = float(dc_cfg["drift_winrate_threshold"])

    run_id     = context.get("run_id", str(uuid.uuid4())[:8])
    check_time = datetime.now(timezone.utc)

    try:
        # ── 1. Pull recent paper trades ──────────────────────────────────────
        try:
            trades_df = con.execute(f"""
                SELECT net_return, pnl_dollars, specialist_pred_proba, entry_time, exit_reason
                FROM paper_trade_agent_log
                ORDER BY entry_time DESC
                LIMIT {eval_window}
            """).df()
        except Exception:
            trades_df = pd.DataFrame()

        # ── 2. Insufficient trades guard ─────────────────────────────────────
        if len(trades_df) < min_trades:
            result = {
                "status":            "success",
                "drift_detected":    False,
                "current_pf":        None,
                "baseline_pf":       baseline_pf,
                "current_win_rate":  None,
                "baseline_win_rate": baseline_win_rate,
                "trades_evaluated":  int(len(trades_df)),
                "action_taken":      "no_action_insufficient_trades",
                "message":           f"Only {len(trades_df)} trades, need {min_trades} minimum",
            }
            _log_and_close(con, run_id, check_time, result, baseline_pf, baseline_win_rate)
            return result

        # ── 3. Compute current metrics ───────────────────────────────────────
        wins         = trades_df[trades_df["net_return"] > 0]
        losses       = trades_df[trades_df["net_return"] <= 0]
        gross_profit = float(wins["pnl_dollars"].sum())
        gross_loss   = float(abs(losses["pnl_dollars"].sum()))

        if gross_loss == 0 and gross_profit > 0:
            current_pf = 999.0
        elif gross_loss == 0:
            current_pf = 0.0
        else:
            current_pf = gross_profit / gross_loss

        current_win_rate = float((trades_df["net_return"] > 0).mean())

        # ── 4. Drift detection ───────────────────────────────────────────────
        pf_drift       = bool(current_pf < (baseline_pf * pf_threshold))
        wr_drift       = bool(current_win_rate < wr_threshold)
        drift_detected = bool(pf_drift or wr_drift)

        drift_reasons = []
        if pf_drift:
            drift_reasons.append(
                f"PF {current_pf:.3f} < threshold {baseline_pf * pf_threshold:.3f} "
                f"(baseline {baseline_pf:.3f} x {pf_threshold})"
            )
        if wr_drift:
            drift_reasons.append(
                f"Win rate {current_win_rate:.3f} < floor {wr_threshold:.3f}"
            )

        # ── 5. Action and context update ─────────────────────────────────────
        if drift_detected:
            action_taken               = "retrain_triggered"
            context["drift_detected"]  = True
            print(f"[DRIFT] Drift detected — {' | '.join(drift_reasons)}")
            print(f"[DRIFT] Action: retrain_subagent will be spawned")
        else:
            action_taken               = "no_action_model_healthy"
            context["drift_detected"]  = False
            print(
                f"[DRIFT] Model healthy — "
                f"PF={current_pf:.3f} (baseline={baseline_pf:.3f}) | "
                f"WinRate={current_win_rate:.3f} (baseline={baseline_win_rate:.3f})"
            )

        result = {
            "status":            "success",
            "drift_detected":    drift_detected,
            "current_pf":        round(current_pf, 4),
            "baseline_pf":       baseline_pf,
            "current_win_rate":  round(current_win_rate, 4),
            "baseline_win_rate": baseline_win_rate,
            "trades_evaluated":  int(len(trades_df)),
            "pf_drift":          pf_drift,
            "wr_drift":          wr_drift,
            "drift_reasons":     drift_reasons,
            "action_taken":      action_taken,
        }

        _log_and_close(con, run_id, check_time, result, baseline_pf, baseline_win_rate)
        return result

    except Exception as e:
        con.close()
        return {
            "status":         "failed",
            "drift_detected": False,
            "error":          traceback.format_exc(),
            "action_taken":   "no_action_error",
        }


def _log_and_close(con, run_id, check_time, result, baseline_pf, baseline_win_rate):
    try:
        log_drift(
            con               = con,
            run_id            = run_id,
            check_time        = check_time,
            current_pf        = float(result.get("current_pf") or 0.0),
            baseline_pf       = float(baseline_pf),
            current_win_rate  = float(result.get("current_win_rate") or 0.0),
            baseline_win_rate = float(baseline_win_rate),
            drift_detected    = bool(result.get("drift_detected", False)),
            trades_evaluated  = int(result.get("trades_evaluated", 0)),
            action_taken      = str(result.get("action_taken", "unknown")),
        )
    except Exception as log_err:
        print(f"[WARN] drift log write failed: {log_err}")
    finally:
        con.close()
