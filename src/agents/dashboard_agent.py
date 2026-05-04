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
    output_path = PROJECT_ROOT / dashcfg["output_path"]
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
        latest_trade_df = _safe_df(con, f"""
            SELECT MAX(COALESCE(TRY_CAST(trade_time_utc AS TIMESTAMP), entry_time)) AS t
            FROM {trade_table}
        """)
        latest_run_df = _safe_df(con, f"SELECT MAX(run_start_utc) AS t FROM {run_log_table}")

        hours_stale_market = _stale_hours(latest_market_df)
        hours_stale_pred = _stale_hours(latest_pred_df)
        hours_stale_trade = _stale_hours(latest_trade_df)

        wf_df = _safe_df(con, f"SELECT * FROM {compression_selected} LIMIT 1")

        trades_df = _safe_df(con, f"""
            SELECT
                run_id,
                entry_time,
                exit_time,
                regime_label,
                entry_price,
                exit_price,
                specialist_pred_proba,
                specialist_pred_label,
                vol_20,
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
                trade_time_utc
            FROM {trade_table}
            ORDER BY COALESCE(TRY_CAST(trade_time_utc AS TIMESTAMP), entry_time) DESC
            LIMIT {lookback_trades * 8}
        """)

        runlog_df = _safe_df(con, f"""
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
        """)

        signallog_df = _safe_df(con, f"""
            SELECT
                open_time,
                rel_score,
                close,
                regime,
                emitted_at
            FROM {signal_log_table}
            ORDER BY emitted_at DESC
            LIMIT 10
        """)

        market_df = _safe_df(con, f"""
            SELECT open_time, close
            FROM {market_table}
            ORDER BY open_time DESC
            LIMIT 96
        """)

        latest_pred_row_df = _safe_df(con, f"""
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
        """)

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

        if not trades_df.empty:
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], errors="coerce")
            trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], errors="coerce")

            if "trade_time_utc" in trades_df.columns:
                trades_df["trade_written_at"] = pd.to_datetime(
                    trades_df["trade_time_utc"], errors="coerce", utc=True
                ).dt.tz_convert(None)
            else:
                trades_df["trade_written_at"] = pd.NaT

            trades_df["sort_ts"] = trades_df["trade_written_at"].fillna(trades_df["entry_time"])
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
                    start_eq = float(cfg["paper_trade_agent"]["initial_equity"])
                    eq_df["ret_pct"] = (eq_df["account_equity"] / start_eq - 1.0) * 100.0
                    equity_labels = [_fmt_ts(t) for t in eq_df["sort_ts"].tolist()]
                    equity_values = [round(float(x), 3) for x in eq_df["ret_pct"].tolist()]
                    total_return = float(eq_df["account_equity"].iloc[-1] / start_eq - 1.0)

                view_df = trades_df.sort_values("sort_ts", ascending=False).head(lookback_trades)
                for _, r in view_df.iterrows():
                    net_ret_pct = float(pd.to_numeric(pd.Series([r.get("net_return")]), errors="coerce").fillna(0.0).iloc[0]) * 100.0
                    pnl_val = float(pd.to_numeric(pd.Series([r.get("pnl_dollars")]), errors="coerce").fillna(0.0).iloc[0])
                    recent_trade_rows.append(
                        {
                            "entry_time": _fmt_ts(r.get("entry_time")),
                            "entry_price": _fmt_num(r.get("entry_price"), 2),
                            "exit_price": _fmt_num(r.get("exit_price"), 2),
                            "confidence": _fmt_num(r.get("specialist_pred_proba"), 3),
                            "exit_reason": r.get("exit_reason", ""),
                            "net_return": round(net_ret_pct, 3),
                            "pnl": round(pnl_val, 2),
                            "bars_held": int(pd.to_numeric(pd.Series([r.get("bars_held")]), errors="coerce").fillna(0).iloc[0]),
                            "written_at": _fmt_ts(r.get("sort_ts")),
                        }
                    )

        run_rows_html = ""
        if not runlog_df.empty:
            for _, r in runlog_df.head(10).iterrows():
                status_html = (
                    '<span class="pill pill-red">ABORTED</span>'
                    if bool(r.get("aborted"))
                    else '<span class="pill pill-green">OK</span>'
                )
                drift_html = (
                    '<span class="pill pill-yellow">DRIFT</span>'
                    if bool(r.get("drift_detected"))
                    else '<span class="pill pill-gray">NO</span>'
                )
                retrain_html = (
                    '<span class="pill pill-purple">YES</span>'
                    if bool(r.get("retrained"))
                    else '<span class="pill pill-gray">NO</span>'
                )
                run_rows_html += f"""
                <tr>
                  <td class="mono">{_fmt_ts(r.get('run_start_utc'))}</td>
                  <td class="mono muted">{str(r.get('run_id', ''))[:8]}</td>
                  <td>{status_html}</td>
                  <td class="mono">{int(r.get('new_rows_added', 0) or 0)}</td>
                  <td>{drift_html}</td>
                  <td>{retrain_html}</td>
                  <td class="mono">{int(r.get('signals_emitted', 0) or 0)}</td>
                </tr>
                """

        trade_rows_html = ""
        if recent_trade_rows:
            for row in recent_trade_rows:
                ret_color = "var(--green)" if float(row["net_return"]) >= 0 else "var(--red)"
                pnl_color = "var(--green)" if float(row["pnl"]) >= 0 else "var(--red)"
                trade_rows_html += f"""
                <tr>
                  <td>{row['entry_time']}</td>
                  <td class="mono">{row['entry_price']}</td>
                  <td class="mono">{row['exit_price']}</td>
                  <td><span class="pill pill-blue">{row['confidence']}</span></td>
                  <td><span class="pill pill-gray">{row['exit_reason']}</span></td>
                  <td class="mono" style="color:{ret_color}">{row['net_return']}</td>
                  <td class="mono" style="color:{pnl_color}">{row['pnl']:.2f}</td>
                  <td class="mono">{row['bars_held']}</td>
                </tr>
                """

        signal_rows_html = ""
        latest_signal = None
        if not signallog_df.empty:
            signallog_df["emitted_at"] = pd.to_datetime(signallog_df["emitted_at"], errors="coerce")
            signallog_df["open_time"] = pd.to_datetime(signallog_df["open_time"], errors="coerce")
            signallog_df = signallog_df.sort_values("emitted_at", ascending=False)
            latest_signal = signallog_df.iloc[0].to_dict()
            for _, r in signallog_df.head(10).iterrows():
                rel_score_val = float(pd.to_numeric(pd.Series([r.get("rel_score")]), errors="coerce").fillna(0.0).iloc[0])
                signal_rows_html += f"""
                <tr>
                  <td>{_fmt_ts(r.get('emitted_at'))}</td>
                  <td>{_fmt_ts(r.get('open_time'))}</td>
                  <td><span class="pill pill-green">{_fmt_num(rel_score_val * 100.0, 1)}</span></td>
                  <td class="mono">{_fmt_num(r.get('close'), 2)}</td>
                  <td><span class="pill pill-blue">{r.get('regime', '')}</span></td>
                </tr>
                """

        latest_pred = None
        if not latest_pred_row_df.empty:
            latest_pred = latest_pred_row_df.iloc[0].to_dict()

        price_labels_js = "[]"
        price_now_js = "[]"
        price_future_js = "[]"
        if not market_df.empty:
            mdf = market_df.copy()
            mdf["open_time"] = pd.to_datetime(mdf["open_time"], errors="coerce")
            mdf["close"] = pd.to_numeric(mdf["close"], errors="coerce")
            mdf = mdf.dropna().sort_values("open_time")
            if not mdf.empty:
                labels = [t.strftime("%Y-%m-%d %H:%M") for t in mdf["open_time"].tolist()]
                closes = [round(float(x), 2) for x in mdf["close"].tolist()]
                price_labels_js = json.dumps(labels)
                price_now_js = json.dumps(closes)

                if len(closes) >= 4:
                    last = closes[-1]
                    delta = closes[-1] - closes[-4]
                    future_path = [None] * (len(closes) - 1) + [
                        round(last, 2),
                        round(last + delta * 0.45, 2),
                        round(last + delta * 0.85, 2),
                    ]
                    price_future_js = json.dumps(future_path)
                else:
                    price_future_js = json.dumps([])

        suggestion = "No action"
        suggestion_detail = "Model is neutral — no fresh high-confidence setup."
        option_type = "NONE"
        side = "NONE"
        strike_hint = "N/A"
        target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        confidence_pct = 0.0
        source_label = "latest prediction"

        if latest_pred is not None:
            rel = float(pd.to_numeric(pd.Series([latest_pred.get("rel_score")]), errors="coerce").fillna(0.0).iloc[0])
            price = float(pd.to_numeric(pd.Series([latest_pred.get("close")]), errors="coerce").fillna(0.0).iloc[0])
            regime = str(latest_pred.get("regime", "compression"))
            confidence_pct = rel * 100.0
            strike_hint = f"{round(price):.0f}"

            if rel >= 0.90:
                suggestion = "CALL BUY"
                option_type = "CALL"
                side = "BUY"
                suggestion_detail = f"Strong upside setup in {regime}. Bias remains constructive above ~{price:.0f} for the next 24–48h."
            elif rel >= 0.78:
                suggestion = "LIGHT CALL BUY"
                option_type = "CALL"
                side = "BUY"
                suggestion_detail = f"Qualified upside signal in {regime}. Consider lighter sizing around ~{price:.0f} with disciplined risk."
            elif rel <= 0.20:
                suggestion = "PUT BUY / HEDGE"
                option_type = "PUT"
                side = "BUY"
                suggestion_detail = f"Downside pressure elevated in {regime}. Hedge or bearish bias near ~{price:.0f}."
            else:
                suggestion = "WATCH"
                option_type = "NONE"
                side = "WAIT"
                suggestion_detail = f"Latest prediction is below action threshold in {regime}. Monitor rather than force a trade."

        def _fresh_label(hours):
            if hours is None:
                return "N/A"
            if hours < 2:
                return "LIVE"
            return f"STALE {hours:.1f}h"

        def _trade_label(hours, has_recent_rows):
            if has_recent_rows:
                if hours is not None and hours < 24:
                    return "LIVE"
                if hours is None:
                    return "RECENT"
                return f"STALE {hours:.1f}h"
            return "NO RECENT TRADES"

        market_status = _fresh_label(hours_stale_market)
        pred_status = _fresh_label(hours_stale_pred)
        trade_status = _trade_label(hours_stale_trade, not trades_df.empty)

        lastrun = ""
        if not latest_run_df.empty and not pd.isna(latest_run_df.iloc[0]["t"]):
            lastrun = _fmt_ts(latest_run_df.iloc[0]["t"])

        ret_color = "green" if total_return >= 0 else "red"
        wr_color = "green" if win_rate >= 0.55 else "yellow"
        pf_color = "green" if oos_pf >= 1.5 else "yellow"
        drft_color = "yellow" if drift_events > 0 else "green"

        equity_labels_js = json.dumps(equity_labels)
        equity_values_js = json.dumps(equity_values)

        html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Claudex ETHUSD 1H</title>
  <meta http-equiv="refresh" content="3600">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root,
    [data-theme="dark"] {{
      --bg: #0b0d13;
      --surface: #11131b;
      --surface-2: #151925;
      --surface-3: #1c2233;
      --border: rgba(98, 126, 234, 0.24);
      --border-2: rgba(138, 146, 178, 0.16);

      --text: #eef2ff;
      --muted: #a9b3d1;
      --faint: #5f6885;

      --blue: #627eea;
      --green: #3ddc97;
      --red: #ff6b81;
      --yellow: #f5c45b;
      --purple: #8a92b2;

      --header-glow: rgba(98, 126, 234, 0.18);
      --header-glow-2: rgba(138, 146, 178, 0.12);

      --font-body: "Inter", -apple-system, system-ui, sans-serif;
      --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --radius: 8px;
      --radius-lg: 14px;
    }}

    [data-theme="light"] {{
      --bg: #f3f6fd;
      --surface: #ffffff;
      --surface-2: #eef2fb;
      --surface-3: #e6ebf8;
      --border: rgba(98, 126, 234, 0.20);
      --border-2: rgba(98, 126, 234, 0.12);

      --text: #1c2540;
      --muted: #62708e;
      --faint: #96a2bf;

      --blue: #4f6de6;
      --green: #1fa971;
      --red: #dd4f68;
      --yellow: #d69b2d;
      --purple: #7f87aa;

      --header-glow: rgba(98, 126, 234, 0.10);
      --header-glow-2: rgba(138, 146, 178, 0.08);
    }}

    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html {{
      color-scheme: dark light;
    }}

    body {{
      font-family: var(--font-body);
      background:
        radial-gradient(circle at top, var(--header-glow) 0%, transparent 28%),
        radial-gradient(circle at top right, var(--header-glow-2) 0%, transparent 22%),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
      transition: background 180ms ease, color 180ms ease;
    }}

    .app {{
      max-width: 1180px;
      margin: 16px auto 32px auto;
      padding: 0 16px 32px 16px;
    }}

    .header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      border-radius: var(--radius-lg);
      background: linear-gradient(
        120deg,
        color-mix(in srgb, var(--blue) 14%, transparent),
        color-mix(in srgb, var(--purple) 12%, transparent)
      );
      border: 1px solid var(--border);
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.28);
      margin-bottom: 18px;
      transition: background 180ms ease, border-color 180ms ease;
    }}

    .header-left h1 {{
      font-size: 1.15rem;
      letter-spacing: 0.03em;
    }}

    .header-left p {{
      margin-top: 4px;
      font-size: 0.75rem;
      color: var(--muted);
    }}

    .header-right {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    .model-badge {{
      font-size: 0.72rem;
      padding: 5px 9px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--surface) 92%, transparent);
      border: 1px solid var(--border);
      color: var(--text);
    }}

    .status-label {{
      font-size: 0.72rem;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    .status-dot {{
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--green);
      box-shadow: 0 0 8px color-mix(in srgb, var(--green) 70%, transparent);
      animation: pulse 1.4s ease-in-out infinite;
    }}

    @keyframes pulse {{
      0% {{ transform: scale(1); opacity: 0.9; }}
      50% {{ transform: scale(1.6); opacity: 0.15; }}
      100% {{ transform: scale(1); opacity: 0.9; }}
    }}

    .theme-toggle {{
      background: color-mix(in srgb, var(--surface) 88%, transparent);
      border: 1px solid var(--border);
      border-radius: 999px;
      width: 34px;
      height: 34px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      color: var(--muted);
      transition: background 150ms ease, color 150ms ease, transform 150ms ease, border-color 150ms ease;
    }}

    .theme-toggle:hover {{
      transform: translateY(-1px);
      color: var(--text);
      background: color-mix(in srgb, var(--surface-3) 95%, transparent);
    }}

    .kpi-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }}

    .kpi {{
      flex: 1 1 140px;
      min-width: 120px;
      padding: 12px 13px;
      border-radius: var(--radius);
      border: 1px solid var(--border);
      background: linear-gradient(180deg, var(--surface-2), var(--surface));
      position: relative;
      overflow: hidden;
      transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
    }}

    .kpi:hover {{
      transform: translateY(-1px);
    }}

    .kpi::after {{
      content: "";
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at top left, color-mix(in srgb, var(--blue) 18%, transparent), transparent 58%);
      opacity: 0;
      transition: opacity 180ms ease-out;
      pointer-events: none;
    }}

    .kpi:hover::after {{
      opacity: 1;
    }}

    .kpi-value {{
      font-size: 1.4rem;
      font-weight: 700;
      line-height: 1.1;
    }}

    .kpi-label {{
      margin-top: 4px;
      font-size: 0.7rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    .grid {{
      display: grid;
      grid-template-columns: 2fr 1.6fr;
      gap: 16px;
    }}

    @media (max-width: 900px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .header-right {{
        width: 100%;
        justify-content: flex-start;
      }}
    }}

    .panel,
    .chart-card,
    .table-card {{
      border-radius: var(--radius-lg);
      border: 1px solid var(--border);
      background: linear-gradient(180deg, var(--surface-2), var(--surface));
      transition: background 160ms ease, border-color 160ms ease;
    }}

    .panel {{
      padding: 12px 14px;
      margin-bottom: 16px;
    }}

    .chart-card {{
      padding: 10px 12px 4px 12px;
      height: 230px;
    }}

    .table-card {{
      padding: 10px 12px;
    }}

    .panel-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}

    .panel-title {{
      font-size: 0.82rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}

    .panel-tag {{
      font-size: 0.7rem;
      color: var(--muted);
    }}

    .guidance-main {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }}

    .guidance-pill {{
      font-size: 0.8rem;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: radial-gradient(circle at top, color-mix(in srgb, var(--blue) 18%, transparent), transparent 62%);
      color: var(--text);
    }}

    .guidance-detail {{
      font-size: 0.78rem;
      color: var(--text);
      max-width: 72ch;
    }}

    .guidance-meta {{
      margin-top: 5px;
      font-size: 0.72rem;
      color: var(--muted);
      line-height: 1.45;
    }}

    .chart-container {{
      position: relative;
      width: 100%;
      height: 170px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.76rem;
    }}

    thead tr {{
      background: color-mix(in srgb, var(--surface-3) 90%, transparent);
    }}

    th {{
      text-align: left;
      padding: 6px 6px;
      color: var(--muted);
      font-weight: 500;
    }}

    td {{
      padding: 6px 6px;
      border-top: 1px solid var(--border-2);
      color: var(--text);
    }}

    tbody tr:hover {{
      background: color-mix(in srgb, var(--surface-3) 82%, transparent);
    }}

    .mono {{
      font-family: var(--font-mono);
      font-feature-settings: "tnum" 1;
    }}

    .muted {{
      color: var(--muted);
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 0.7rem;
      white-space: nowrap;
    }}

    .pill-green {{ background: color-mix(in srgb, var(--green) 14%, transparent); color: var(--green); }}
    .pill-blue {{ background: color-mix(in srgb, var(--blue) 16%, transparent); color: var(--blue); }}
    .pill-gray {{ background: color-mix(in srgb, var(--purple) 14%, transparent); color: var(--muted); }}
    .pill-yellow {{ background: color-mix(in srgb, var(--yellow) 16%, transparent); color: var(--yellow); }}
    .pill-red {{ background: color-mix(in srgb, var(--red) 16%, transparent); color: var(--red); }}
    .pill-purple {{ background: color-mix(in srgb, var(--purple) 16%, transparent); color: var(--purple); }}
  </style>
</head>
<body>
  <div class="app">
    <header class="header">
      <div class="header-left">
        <h1>Claudex ETHUSD 1H</h1>
        <p>Self-improving compression specialist · band {band_name} · Run {run_id}</p>
      </div>
      <div class="header-right">
        <span class="model-badge">{active_model}</span>
        <button class="theme-toggle" data-theme-toggle aria-label="Switch theme"></button>
        <span class="status-label">
          <span class="status-dot"></span>
          Generated · {generated_at}
        </span>
      </div>
    </header>

    <section style="margin-bottom:14px;">
      <div class="kpi-row">
        <div class="kpi">
          <div class="kpi-value" style="color:var(--{ret_color});">{total_return * 100:.2f}%</div>
          <div class="kpi-label">Total Return</div>
        </div>
        <div class="kpi">
          <div class="kpi-value" style="color:var(--{wr_color});">{win_rate * 100:.1f}%</div>
          <div class="kpi-label">Win Rate</div>
        </div>
        <div class="kpi">
          <div class="kpi-value">{total_trades}</div>
          <div class="kpi-label">Recent Trades</div>
        </div>
        <div class="kpi">
          <div class="kpi-value">{signals_total}</div>
          <div class="kpi-label">Signals Emitted</div>
        </div>
        <div class="kpi">
          <div class="kpi-value" style="color:var(--{pf_color});">{oos_pf:.2f}</div>
          <div class="kpi-label">WF Profit Factor</div>
        </div>
        <div class="kpi">
          <div class="kpi-value" style="color:var(--{drft_color});">{drift_events}</div>
          <div class="kpi-label">Drift Events</div>
        </div>
      </div>
    </section>

    <section>
      <div class="grid">
        <div>
          <div class="panel">
            <div class="panel-header">
              <span class="panel-title">Trade Guidance · Model View</span>
              <span class="panel-tag">Prediction-driven summary</span>
            </div>
            <div class="guidance-main">
              <span class="guidance-pill mono">{suggestion}</span>
              <div class="guidance-detail">{suggestion_detail}</div>
            </div>
            <div class="guidance-meta">
              Option type <span class="mono">{option_type}</span> ·
              Side <span class="mono">{side}</span> ·
              Strike ref <span class="mono">{strike_hint}</span> ·
              Target date <span class="mono">{target_date}</span> ·
              Confidence <span class="mono">{confidence_pct:.1f}%</span>
            </div>
            <div class="guidance-meta">
              Market <span class="mono">{market_status}</span> ·
              Predictions <span class="mono">{pred_status}</span> ·
              Trades <span class="mono">{trade_status}</span> ·
              Last run <span class="mono">{lastrun}</span> ·
              Source <span class="mono">{source_label}</span>
            </div>
          </div>

          <div class="chart-card">
            <div class="panel-header" style="margin-bottom:4px;">
              <span class="panel-title">Equity Curve · Recent Paper Account</span>
              <span class="panel-tag">Base {cfg["paper_trade_agent"]["initial_equity"]:,} · last {recent_trade_days}d</span>
            </div>
            <div class="chart-container">
              <canvas id="equityChart"></canvas>
            </div>
          </div>

          <div class="table-card" style="margin-top:12px;">
            <div class="panel-header">
              <span class="panel-title">Recent Trades · Last {lookback_trades}</span>
              <span class="panel-tag">Filtered to recent write activity</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Entry</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>Conf</th>
                  <th>Exit</th>
                  <th>Return %</th>
                  <th>PnL</th>
                  <th>Hold</th>
                </tr>
              </thead>
              <tbody>
                {"<tr><td colspan='8' class='muted' style='text-align:center;padding:10px;'>No recent trades</td></tr>" if not trade_rows_html else trade_rows_html}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <div class="chart-card">
            <div class="panel-header" style="margin-bottom:4px;">
              <span class="panel-title">ETHUSD · Nowcast + Hypothetical</span>
              <span class="panel-tag">Last 96h closes · dashed line = hypothetical</span>
            </div>
            <div class="chart-container">
              <canvas id="priceChart"></canvas>
            </div>
          </div>

          <div class="table-card" style="margin-top:12px;">
            <div class="panel-header">
              <span class="panel-title">Recent Signals</span>
              <span class="panel-tag">Latest 10 · rel_score band ≥ 0.78</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Emitted</th>
                  <th>Bar Open</th>
                  <th>Rel%</th>
                  <th>Close</th>
                  <th>Regime</th>
                </tr>
              </thead>
              <tbody>
                {"<tr><td colspan='5' class='muted' style='text-align:center;padding:10px;'>No signals yet</td></tr>" if not signal_rows_html else signal_rows_html}
              </tbody>
            </table>
          </div>

          <div class="table-card" style="margin-top:12px;">
            <div class="panel-header">
              <span class="panel-title">Orchestrator Runs</span>
              <span class="panel-tag">Last 10 runs</span>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Started UTC</th>
                  <th>Run ID</th>
                  <th>Status</th>
                  <th>New</th>
                  <th>Drift</th>
                  <th>Retrain</th>
                  <th>Signals</th>
                </tr>
              </thead>
              <tbody>
                {"<tr><td colspan='7' class='muted' style='text-align:center;padding:10px;'>No runs logged yet</td></tr>" if not run_rows_html else run_rows_html}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    (function () {{
      const root = document.documentElement;
      const toggle = document.querySelector("[data-theme-toggle]");

      const sunIcon = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="4"></circle>
          <path d="M12 2v2"></path>
          <path d="M12 20v2"></path>
          <path d="m4.93 4.93 1.41 1.41"></path>
          <path d="m17.66 17.66 1.41 1.41"></path>
          <path d="M2 12h2"></path>
          <path d="M20 12h2"></path>
          <path d="m6.34 17.66-1.41 1.41"></path>
          <path d="m19.07 4.93-1.41 1.41"></path>
        </svg>`;

      const moonIcon = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9z"></path>
        </svg>`;

      let theme = (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches)
        ? "light"
        : "dark";

      function applyTheme() {{
        root.setAttribute("data-theme", theme);
        if (!toggle) return;
        toggle.innerHTML = theme === "dark" ? sunIcon : moonIcon;
        toggle.setAttribute(
          "aria-label",
          "Switch to " + (theme === "dark" ? "light" : "dark") + " mode"
        );
      }}

      applyTheme();

      if (toggle) {{
        toggle.addEventListener("click", function () {{
          theme = theme === "dark" ? "light" : "dark";
          applyTheme();
        }});
      }}
    }})();

    (function () {{
      const labels = {equity_labels_js};
      const values = {equity_values_js};
      const ctx = document.getElementById("equityChart");
      if (!ctx) return;

      if (!labels.length) {{
        ctx.parentElement.innerHTML = '<div class="muted" style="padding:24px 8px;">No recent trade-equity points available.</div>';
        return;
      }}

      new Chart(ctx, {{
        type: "line",
        data: {{
          labels,
          datasets: [{{
            label: "Equity Return %",
            data: values,
            borderColor: "#627eea",
            backgroundColor: "rgba(98,126,234,0.12)",
            borderWidth: 2,
            pointRadius: 1.6,
            fill: true,
            tension: 0.35,
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              callbacks: {{
                label: function(context) {{
                  return "Return: " + context.parsed.y.toFixed(2) + "%";
                }}
              }}
            }}
          }},
          scales: {{
            x: {{
              ticks: {{ color: getComputedStyle(document.documentElement).getPropertyValue("--muted"), maxTicksLimit: 6 }},
              grid: {{ color: "rgba(127,135,170,0.18)" }},
            }},
            y: {{
              ticks: {{
                color: getComputedStyle(document.documentElement).getPropertyValue("--muted"),
                callback: function(v) {{ return v + "%"; }}
              }},
              grid: {{ color: "rgba(127,135,170,0.18)" }},
            }},
          }},
        }},
      }});
    }})();

    (function () {{
      const labels = {price_labels_js};
      const nowData = {price_now_js};
      const futureData = {price_future_js};
      const ctx = document.getElementById("priceChart");
      if (!ctx || !labels.length) return;

      const extendedLabels = [...labels, "T+1", "T+2"];
      const alignedNow = [...nowData, null, null];

      new Chart(ctx, {{
        type: "line",
        data: {{
          labels: extendedLabels,
          datasets: [
            {{
              label: "ETH Close",
              data: alignedNow,
              borderColor: "#3ddc97",
              backgroundColor: "rgba(61,220,151,0.08)",
              borderWidth: 2,
              pointRadius: 0,
              fill: true,
              tension: 0.35,
            }},
            {{
              label: "Hypothetical",
              data: futureData,
              borderColor: "#f5c45b",
              borderWidth: 1.6,
              borderDash: [6, 4],
              pointRadius: 0,
              fill: false,
              tension: 0.35,
            }},
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              mode: "index",
              intersect: false,
            }}
          }},
          interaction: {{
            mode: "index",
            intersect: false
          }},
          scales: {{
            x: {{
              ticks: {{ color: getComputedStyle(document.documentElement).getPropertyValue("--muted"), maxTicksLimit: 7 }},
              grid: {{ color: "rgba(127,135,170,0.18)" }},
            }},
            y: {{
              ticks: {{ color: getComputedStyle(document.documentElement).getPropertyValue("--muted") }},
              grid: {{ color: "rgba(127,135,170,0.18)" }},
            }},
          }},
        }},
      }});
    }})();
  </script>
</body>
</html>
"""
        output_path.write_text(html, encoding="utf-8")

        return {
            "status": "success",
            "dashboard_path": str(output_path),
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
            "dashboard_path": "",
        }
    finally:
        con.close()