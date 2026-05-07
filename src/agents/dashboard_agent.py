import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _safe_df(con, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).df()
    except Exception:
        return pd.DataFrame()


def _fmt_ts(x):
    if pd.isna(x) or x is None:
        return ""
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(x)


def _fmt_num(x, nd=2):
    if pd.isna(x) or x is None:
        return ""
    try:
        return round(float(x), nd)
    except Exception:
        return x


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=bool)
    return s.fillna(False).astype(bool)


def run_dashboard_agent(cfg: dict, context: dict) -> dict:
    dashcfg = cfg["dashboard_agent"]
    paths = cfg["paths"]
    tables = cfg["tables"]

    db_path = PROJECT_ROOT / paths["db_path"]
    output_path = PROJECT_ROOT / "src/dashboard/claudex_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))
    try:
        run_id = context.get("run_id", "unknown")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        active_model = Path(cfg["prediction_agent"]["model_path"]).name
        band_name = cfg["paper_trade_agent"].get("band_name", "ge_0.78")
        lookback_trades = int(dashcfg.get("lookback_trades", 50))
        recent_trade_days = int(dashcfg.get("recent_trade_days", 14))

        market_table = tables["canonical_market"]
        pred_table = tables["predictions"]
        trade_table = tables["paper_trades"]
        run_log_table = tables["run_log"]
        signal_log_table = tables["signal_log"]
        compression_selected = tables["compression_selected"]

        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)

        def _stale_hours(df, col="t"):
            if df.empty or col not in df.columns or pd.isna(df.iloc[0][col]):
                return None
            ts = pd.to_datetime(df.iloc[0][col], errors="coerce")
            if pd.isna(ts):
                return None
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
            return (now_utc - ts).total_seconds() / 3600.0

        latest_market_df = _safe_df(con, f"SELECT MAX(open_time) AS t FROM {market_table}")
        latest_pred_df = _safe_df(con, f"SELECT MAX(open_time) AS t FROM {pred_table}")
        latest_trade_df = _safe_df(
            con,
            f"""
            SELECT MAX(COALESCE(TRY_CAST(trade_time_utc AS TIMESTAMP), entry_time)) AS t
            FROM {trade_table}
            """
        )
        latest_run_df = _safe_df(con, f"SELECT MAX(run_start_utc) AS t FROM {run_log_table}")

        hours_stale_market = _stale_hours(latest_market_df)
        hours_stale_pred = _stale_hours(latest_pred_df)
        hours_stale_trade = _stale_hours(latest_trade_df)

        wf_df = _safe_df(con, f"SELECT * FROM {compression_selected} LIMIT 1")

        trades_df = _safe_df(
            con,
            f"""
            SELECT
                run_id,
                entry_time,
                exit_time,
                regime_label,
                entry_price,
                exit_price,
                specialist_pred_proba,
                specialist_pred_label,
                TRY_CAST(vol20 AS DOUBLE) AS vol20,
                stop_pct,
                take_profit_pct,
                exit_reason,
                bars_held,
                breakeven_activated,
                partial_tp_hit,
                gross_return,
                net_return,
                pnl_dollars,
                account_equity,
                trade_time_utc,
                COALESCE(
                    TRY_CAST(trade_time_utc AS TIMESTAMP),
                    CAST(entry_time AS TIMESTAMP)
                ) AS sort_ts
            FROM {trade_table}
            ORDER BY sort_ts DESC
            LIMIT {lookback_trades * 8}
            """
        )

        runlog_df = _safe_df(
            con,
            f"""
            SELECT
                run_id,
                run_start_utc,
                aborted,
                new_rows_added,
                drift_detected,
                retrained,
                signals_emitted
            FROM {run_log_table}
            ORDER BY run_start_utc DESC
            LIMIT 20
            """
        )

        signallog_df = _safe_df(
            con,
            f"""
            SELECT
                open_time,
                rel_score,
                close,
                regime,
                emitted_at
            FROM {signal_log_table}
            ORDER BY emitted_at DESC
            LIMIT 10
            """
        )

        market_df = _safe_df(
            con,
            f"""
            SELECT open_time, close
            FROM {market_table}
            ORDER BY open_time DESC
            LIMIT 96
            """
        )

        latest_pred_row_df = _safe_df(
            con,
            f"""
            SELECT
                open_time,
                close,
                rel_score,
                proba,
                cal_proba,
                regime,
                pred_label
            FROM {pred_table}
            ORDER BY open_time DESC
            LIMIT 1
            """
        )

        drift_events = 0
        signals_total = 0
        if not runlog_df.empty and "drift_detected" in runlog_df.columns:
            drift_events = int(_coerce_bool_series(runlog_df["drift_detected"]).sum())
        if not signallog_df.empty:
            signals_total = len(signallog_df)

        oos_pf = 0.0
        oos_total_return = 0.0
        oos_win_rate = 0.0
        if not wf_df.empty:
            wfrow = wf_df.iloc[0].to_dict()
            oos_pf = float(wfrow.get("oos_profit_factor", wfrow.get("profit_factor", 0.0)) or 0.0)
            oos_total_return = float(wfrow.get("oos_total_net_return", wfrow.get("oos_return", 0.0)) or 0.0)
            oos_win_rate = float(wfrow.get("oos_win_rate", wfrow.get("win_rate", 0.0)) or 0.0)

        total_trades = 0
        win_rate = 0.0
        total_return = 0.0
        equity_labels = []
        equity_values = []
        recent_trade_rows = []

        start_eq = float(cfg["paper_trade_agent"]["initial_equity"])
        current_eq = start_eq
        total_pnl = 0.0

        if not trades_df.empty:
            trades_df["sort_ts"] = pd.to_datetime(trades_df["sort_ts"], errors="coerce")
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], errors="coerce")
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], errors="coerce")

            cutoff = now_utc - pd.Timedelta(days=recent_trade_days)
            trades_df = trades_df[trades_df["sort_ts"].notna()].copy()
            trades_df = trades_df[trades_df["sort_ts"] >= cutoff].copy()
            trades_df = trades_df.sort_values("sort_ts")
            trades_df = trades_df.drop_duplicates(subset=["entry_time", "exit_time"], keep="last")

            if not trades_df.empty:
                total_trades = len(trades_df)
                nr = pd.to_numeric(trades_df["net_return"], errors="coerce").fillna(0.0)
                win_rate = float((nr > 0).mean())

                eq_df = trades_df[["sort_ts", "account_equity"]].copy()
                eq_df["account_equity"] = pd.to_numeric(eq_df["account_equity"], errors="coerce")
                eq_df = eq_df.dropna()

                if not eq_df.empty:
                    current_eq = float(eq_df["account_equity"].iloc[-1])
                    total_pnl = current_eq - start_eq
                    eq_df["ret_pct"] = (eq_df["account_equity"] / start_eq - 1.0) * 100.0
                    equity_labels = [_fmt_ts(t) for t in eq_df["sort_ts"].tolist()]
                    equity_values = [round(float(x), 3) for x in eq_df["ret_pct"].tolist()]
                    total_return = float(current_eq / start_eq - 1.0)

                view_df = trades_df.sort_values("sort_ts", ascending=False).head(lookback_trades)
                for _, r in view_df.iterrows():
                    net_ret_pct = float(
                        pd.to_numeric(pd.Series([r.get("net_return")]), errors="coerce").fillna(0.0).iloc[0]
                    ) * 100.0
                    pnl_val = float(
                        pd.to_numeric(pd.Series([r.get("pnl_dollars")]), errors="coerce").fillna(0.0).iloc[0]
                    )
                    recent_trade_rows.append(
                        {
                            "entry_time": _fmt_ts(r.get("entry_time")),
                            "exit_time": _fmt_ts(r.get("exit_time")),
                            "entry_price": _fmt_num(r.get("entry_price"), 2),
                            "exit_price": _fmt_num(r.get("exit_price"), 2),
                            "confidence": _fmt_num(r.get("specialist_pred_proba"), 3),
                            "regime_label": r.get("regime_label", ""),
                            "exit_reason": r.get("exit_reason", ""),
                            "net_return_pct": round(net_ret_pct, 3),
                            "pnl": round(pnl_val, 2),
                            "bars_held": int(
                                pd.to_numeric(pd.Series([r.get("bars_held")]), errors="coerce").fillna(0).iloc[0]
                            ),
                            "breakeven_activated": bool(r.get("breakeven_activated")),
                            "partial_tp_hit": bool(r.get("partial_tp_hit")),
                            "gross_return": _fmt_num(r.get("gross_return"), 6),
                            "net_return": _fmt_num(r.get("net_return"), 6),
                            "account_equity": _fmt_num(r.get("account_equity"), 2),
                            "written_at": _fmt_ts(r.get("sort_ts")),
                        }
                    )

        recent_run_rows = []
        if not runlog_df.empty:
            for _, r in runlog_df.head(10).iterrows():
                recent_run_rows.append(
                    {
                        "run_start_utc": _fmt_ts(r.get("run_start_utc")),
                        "run_id": str(r.get("run_id", ""))[:8],
                        "aborted": bool(r.get("aborted")),
                        "status": "ABORTED" if bool(r.get("aborted")) else "OK",
                        "new_rows_added": int(r.get("new_rows_added", 0) or 0),
                        "drift_detected": bool(r.get("drift_detected")),
                        "drift_label": "DRIFT" if bool(r.get("drift_detected")) else "NO",
                        "retrained": bool(r.get("retrained")),
                        "retrained_label": "YES" if bool(r.get("retrained")) else "NO",
                        "signals_emitted": int(r.get("signals_emitted", 0) or 0),
                    }
                )

        recent_signal_rows = []
        latest_signal = None
        if not signallog_df.empty:
            signallog_df["emitted_at"] = pd.to_datetime(signallog_df["emitted_at"], errors="coerce")
            signallog_df["open_time"] = pd.to_datetime(signallog_df["open_time"], errors="coerce")
            signallog_df = signallog_df.sort_values("emitted_at", ascending=False)
            latest_signal = signallog_df.iloc[0].to_dict()

            for _, r in signallog_df.head(10).iterrows():
                rel_score_val = float(
                    pd.to_numeric(pd.Series([r.get("rel_score")]), errors="coerce").fillna(0.0).iloc[0]
                )
                recent_signal_rows.append(
                    {
                        "emitted_at": _fmt_ts(r.get("emitted_at")),
                        "open_time": _fmt_ts(r.get("open_time")),
                        "rel_score": round(rel_score_val, 6),
                        "rel_score_pct": round(rel_score_val * 100.0, 1),
                        "close": _fmt_num(r.get("close"), 2),
                        "regime": r.get("regime", ""),
                    }
                )

        latest_pred = None
        if not latest_pred_row_df.empty:
            latest_pred = latest_pred_row_df.iloc[0].to_dict()

        price_labels = []
        price_now = []
        price_future = []
        if not market_df.empty:
            mdf = market_df.copy()
            mdf["open_time"] = pd.to_datetime(mdf["open_time"], errors="coerce")
            mdf["close"] = pd.to_numeric(mdf["close"], errors="coerce")
            mdf = mdf.dropna().sort_values("open_time")
            if not mdf.empty:
                price_labels = [t.strftime("%Y-%m-%d %H:%M") for t in mdf["open_time"].tolist()]
                closes = [round(float(x), 2) for x in mdf["close"].tolist()]
                price_now = closes
                if len(closes) >= 4:
                    last = closes[-1]
                    delta = closes[-1] - closes[-4]
                    price_future = [None] * (len(closes) - 1) + [
                        round(last, 2),
                        round(last + delta * 0.45, 2),
                        round(last + delta * 0.85, 2),
                    ]

        suggestion = "No action"
        suggestion_detail = "Model is neutral; no fresh high-confidence setup."
        option_type = "NONE"
        side = "NONE"
        strike_hint = "NA"
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        confidence_pct = 0.0
        source_label = "latest prediction"

        latest_prediction_payload = None
        if latest_pred is not None:
            rel = float(pd.to_numeric(pd.Series([latest_pred.get("rel_score")]), errors="coerce").fillna(0.0).iloc[0])
            price = float(pd.to_numeric(pd.Series([latest_pred.get("close")]), errors="coerce").fillna(0.0).iloc[0])
            regime = str(latest_pred.get("regime", "compression"))
            confidence_pct = rel * 100.0
            strike_hint = f"{round(price):.0f}"

            latest_prediction_payload = {
                "open_time": _fmt_ts(latest_pred.get("open_time")),
                "close": round(price, 2),
                "rel_score": round(rel, 6),
                "confidence_pct": round(confidence_pct, 2),
                "proba": _fmt_num(latest_pred.get("proba"), 6),
                "cal_proba": _fmt_num(latest_pred.get("cal_proba"), 6),
                "regime": regime,
                "pred_label": int(pd.to_numeric(pd.Series([latest_pred.get("pred_label")]), errors="coerce").fillna(0).iloc[0]),
            }

            if rel >= 0.90:
                suggestion = "CALL BUY"
                option_type = "CALL"
                side = "BUY"
                suggestion_detail = f"Strong upside setup in {regime}. Bias remains constructive above {price:.0f} for the next 24-48h."
            elif rel >= 0.78:
                suggestion = "LIGHT CALL BUY"
                option_type = "CALL"
                side = "BUY"
                suggestion_detail = f"Qualified upside signal in {regime}. Consider lighter sizing around {price:.0f} with disciplined risk."
            elif rel <= 0.20:
                suggestion = "PUT BUY HEDGE"
                option_type = "PUT"
                side = "BUY"
                suggestion_detail = f"Downside pressure elevated in {regime}. Hedge or bearish bias near {price:.0f}."
            else:
                suggestion = "WATCH"
                option_type = "NONE"
                side = "WAIT"
                suggestion_detail = f"Latest prediction is below action threshold in {regime}. Monitor rather than force a trade."

        def _fresh_label(hours):
            if hours is None:
                return "NA"
            if hours <= 2:
                return "LIVE"
            return f"STALE {hours:.1f}h"

        def _trade_label(hours, has_recent_rows):
            if has_recent_rows:
                if hours is not None and hours <= 24:
                    return "LIVE"
                if hours is None:
                    return "RECENT"
                return f"STALE {hours:.1f}h"
            return "NO RECENT TRADES"

        market_status = _fresh_label(hours_stale_market)
        pred_status = _fresh_label(hours_stale_pred)
        trade_status = _trade_label(hours_stale_trade, not trades_df.empty)

        last_run = ""
        if not latest_run_df.empty and not pd.isna(latest_run_df.iloc[0]["t"]):
            last_run = _fmt_ts(latest_run_df.iloc[0]["t"])

        ret_color = "green" if total_return >= 0 else "red"
        wr_color = "green" if win_rate >= 0.55 else "yellow"
        pf_color = "green" if oos_pf >= 1.5 else "yellow"
        drft_color = "yellow" if drift_events > 0 else "green"
        pnl_color = "green" if total_pnl >= 0 else "red"

        payload = {
            "meta": {
                "title": "Claudex ETHUSD 1H",
                "run_id": run_id,
                "generated_at": generated_at,
                "active_model": active_model,
                "band_name": band_name,
                "lookback_trades": lookback_trades,
                "recent_trade_days": recent_trade_days,
                "trade_table": trade_table,
                "market_table": market_table,
                "pred_table": pred_table,
                "run_log_table": run_log_table,
                "signal_log_table": signal_log_table,
                "compression_selected_table": compression_selected,
            },
            "status": {
                "market_status": market_status,
                "pred_status": pred_status,
                "trade_status": trade_status,
                "last_run": last_run,
                "hours_stale_market": None if hours_stale_market is None else round(float(hours_stale_market), 3),
                "hours_stale_pred": None if hours_stale_pred is None else round(float(hours_stale_pred), 3),
                "hours_stale_trade": None if hours_stale_trade is None else round(float(hours_stale_trade), 3),
            },
            "kpis": {
                "start_equity": round(float(start_eq), 2),
                "current_equity": round(float(current_eq), 2),
                "total_pnl": round(float(total_pnl), 2),
                "total_return": round(float(total_return), 6),
                "total_return_pct": round(float(total_return) * 100.0, 3),
                "win_rate": round(float(win_rate), 6),
                "win_rate_pct": round(float(win_rate) * 100.0, 3),
                "total_trades": int(total_trades),
                "signals_total": int(signals_total),
                "oos_pf": round(float(oos_pf), 6),
                "oos_total_return": round(float(oos_total_return), 6),
                "oos_total_return_pct": round(float(oos_total_return) * 100.0, 3),
                "oos_win_rate": round(float(oos_win_rate), 6),
                "oos_win_rate_pct": round(float(oos_win_rate) * 100.0, 3),
                "drift_events": int(drift_events),
                "colors": {
                    "pnl": pnl_color,
                    "return": ret_color,
                    "win_rate": wr_color,
                    "profit_factor": pf_color,
                    "drift": drft_color,
                },
            },
            "guidance": {
                "suggestion": suggestion,
                "suggestion_detail": suggestion_detail,
                "option_type": option_type,
                "side": side,
                "strike_hint": strike_hint,
                "target_date": target_date,
                "confidence_pct": round(float(confidence_pct), 2),
                "source_label": source_label,
            },
            "latest_prediction": latest_prediction_payload,
            "latest_signal": None if latest_signal is None else {
                "emitted_at": _fmt_ts(latest_signal.get("emitted_at")),
                "open_time": _fmt_ts(latest_signal.get("open_time")),
                "rel_score": round(
                    float(pd.to_numeric(pd.Series([latest_signal.get("rel_score")]), errors="coerce").fillna(0.0).iloc[0]),
                    6,
                ),
                "close": _fmt_num(latest_signal.get("close"), 2),
                "regime": latest_signal.get("regime", ""),
            },
            "equity_curve": {
                "labels": equity_labels,
                "values": equity_values,
                "base_equity": round(float(start_eq), 2),
            },
            "market_chart": {
                "labels": price_labels,
                "now_data": price_now,
                "future_data": price_future,
            },
            "recent_trades": recent_trade_rows,
            "recent_signals": recent_signal_rows,
            "recent_runs": recent_run_rows,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        return {
            "status": "success",
            "data_path": str(output_path),
            "active_model": active_model,
            "trade_table": trade_table,
            "total_trades": total_trades,
            "signals_total": signals_total,
            "generated_at": generated_at,
            "hours_stale_market": hours_stale_market,
            "hours_stale_pred": hours_stale_pred,
            "hours_stale_trade": hours_stale_trade,
            "recent_trade_days": recent_trade_days,
        }

    except Exception:
        return {
            "status": "failed",
            "error": traceback.format_exc(),
            "data_path": "",
        }
    finally:
        con.close()