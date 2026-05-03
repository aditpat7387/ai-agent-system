# =============================================================================
# dashboard_agent.py — Claudex Live Dashboard (animated, guidance, dark/light)
# =============================================================================

import sys
import traceback
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def run_dashboard_agent(cfg: dict, context: dict) -> dict:
    dash_cfg = cfg["dashboard_agent"]
    paths    = cfg["paths"]
    tables   = cfg["tables"]

    db_path     = PROJECT_ROOT / paths["db_path"]
    output_path = PROJECT_ROOT / dash_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))

    try:
        run_id = context.get("run_id", "unknown")
        active_model_path = Path(cfg["prediction_agent"]["model_path"])
        active_model      = active_model_path.name

        # ───────────────────────────────── DATA LOADS ─────────────────────────
        # Live paper trades
        try:
            trades_df = con.execute(f"""
                SELECT DISTINCT entry_time, exit_time, entry_price, exit_price,
                       specialist_pred_proba, exit_reason, net_return,
                       pnl_dollars, account_equity, bars_held
                FROM paper_trade_agent_log
                ORDER BY entry_time DESC
                LIMIT {dash_cfg['lookback_trades']}
            """).df()
        except Exception:
            trades_df = pd.DataFrame()

        # Run log
        try:
            run_log_df = con.execute("""
                SELECT run_id, run_start_utc, aborted, new_rows_added,
                       drift_detected, retrained, signals_emitted
                FROM run_log
                ORDER BY run_start_utc DESC
                LIMIT 20
            """).df()
        except Exception:
            run_log_df = pd.DataFrame()

        # Signal log (new schema: open_time, rel_score, close, regime, emitted_at)
        try:
            signal_log_df = con.execute("""
                SELECT open_time, rel_score, close, regime, emitted_at
                FROM signal_log
                ORDER BY emitted_at DESC
                LIMIT 20
            """).df()
        except Exception:
            signal_log_df = pd.DataFrame()

        # Walk-forward summary for compression band
        try:
            wf_summary_df = con.execute(f"""
                SELECT * FROM {tables['compression_selected']}
            """).df()
        except Exception:
            wf_summary_df = pd.DataFrame()

        # Drift log
        try:
            drift_log_df = con.execute("""
                SELECT drift_detected
                FROM drift_log
                ORDER BY check_time_utc DESC
                LIMIT 10
            """).df()
        except Exception:
            drift_log_df = pd.DataFrame()

        # Recent ETH candles for current + "future" chart
        try:
            market_df = con.execute("""
                SELECT open_time, close
                FROM ethusd_1h_market
                ORDER BY open_time DESC
                LIMIT 96
            """).df()
        except Exception:
            market_df = pd.DataFrame()

        con.close()

        # ───────────────────────────── KPIs & DERIVED STATE ───────────────────
        total_trades = len(trades_df)
        win_rate     = float((trades_df["net_return"] > 0).mean()) if total_trades > 0 else 0.0
        equity       = float(trades_df["account_equity"].iloc[0]) if total_trades > 0 else 100000.0
        total_return = (equity / 100000.0) - 1.0
        signals_total = len(signal_log_df)
        last_run      = str(run_log_df["run_start_utc"].iloc[0])[:16] if len(run_log_df) > 0 else "N/A"
        drift_count   = int(drift_log_df["drift_detected"].sum()) if len(drift_log_df) > 0 else 0

        wf_pf      = float(wf_summary_df["profit_factor"].iloc[0])         if len(wf_summary_df) > 0 else 0.0
        wf_return  = float(wf_summary_df["oos_total_net_return"].iloc[0])   if len(wf_summary_df) > 0 else 0.0
        wf_winrate = float(wf_summary_df["oos_win_rate"].iloc[0])           if len(wf_summary_df) > 0 else 0.0

        # Model "situation" for guidance
        suggestion = "No action"
        suggestion_detail = "Model is neutral; no fresh high-confidence signal."
        option_type = "—"
        side        = "—"
        target_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

        if len(signal_log_df) > 0:
            latest_sig = signal_log_df.iloc[0]
            rel = float(latest_sig["rel_score"])
            price = float(latest_sig["close"])
            regime = str(latest_sig["regime"])
            if rel >= 0.9:
                suggestion = "CALL · BUY"
                option_type = "CALL"
                side = "BUY"
                suggestion_detail = (
                    f"Strong upside compression signal in {regime}. "
                    f"Consider call / long exposure from ~{price:.0f} with next 24–48h horizon."
                )
            elif rel >= 0.8:
                suggestion = "LIGHT CALL · BUY"
                option_type = "CALL"
                side = "BUY"
                suggestion_detail = (
                    f"Moderate upside edge in {regime}. "
                    f"Modest long bias from ~{price:.0f}, risk-managed sizing."
                )
            elif rel <= 0.2:
                suggestion = "PUT · SELL"
                option_type = "PUT"
                side = "SELL"
                suggestion_detail = (
                    f"Downside risk elevated in {regime}. "
                    f"Consider protective puts or short bias around ~{price:.0f}."
                )

        # ── Equity curve data ────────────────────────────────────────────────
        equity_labels_js = "[]"
        equity_values_js = "[]"
        if total_trades > 0:
            eq_df = trades_df.iloc[::-1].copy()
            labels = [str(r["entry_time"])[:16] for _, r in eq_df.iterrows()]
            values = [round((r["account_equity"] / 100000.0 - 1.0) * 100, 3) for _, r in eq_df.iterrows()]
            equity_labels_js = str(labels).replace("'", '"')
            equity_values_js = str(values)

        # ── ETH current + hypothetical near future ───────────────────────────
        price_labels_js = "[]"
        price_now_js    = "[]"
        price_future_js = "[]"
        if len(market_df) > 0:
            mdf = market_df.sort_values("open_time")
            labels = [str(t)[:16] for t in mdf["open_time"]]
            closes = mdf["close"].tolist()
            price_labels_js = str(labels).replace("'", '"')
            price_now_js    = str([round(x, 2) for x in closes])

            # very simple hypothetical: last value extrapolated slightly
            if len(closes) >= 3:
                last = closes[-1]
                prev = closes[-3]
                drift = (last - prev) / max(abs(prev), 1.0)
                future_1 = last * (1 + 0.5 * drift)
                future_2 = last * (1 + 0.9 * drift)
                price_future_js = str([None]*(len(closes)-1) + [round(last,2), round(future_1,2), round(future_2,2)])

        # ───────────────────────────── HTML FRAGMENTS ────────────────────────
        trade_rows = ""
        if total_trades > 0:
            for _, r in trades_df.head(20).iterrows():
                icon    = "✅" if r["net_return"] > 0 else "❌"
                ret_pct = round(r["net_return"] * 100, 3)
                color   = "var(--green)" if r["net_return"] > 0 else "var(--red)"
                trade_rows += f"""
                <tr>
                    <td>{str(r['entry_time'])[:16]}</td>
                    <td class="mono">{round(r['entry_price'], 2)}</td>
                    <td class="mono">{round(r['exit_price'], 2)}</td>
                    <td><span class="pill pill-blue">{round(r['specialist_pred_proba'], 3)}</span></td>
                    <td><span class="pill pill-gray">{r['exit_reason']}</span></td>
                    <td class="mono" style="color:{color}">{icon} {ret_pct:+.3f}%</td>
                    <td class="mono" style="color:{color}">${round(r['pnl_dollars'], 2):+,.2f}</td>
                    <td class="mono">{int(r['bars_held'])}h</td>
                </tr>"""

        signal_rows = ""
        if len(signal_log_df) > 0:
            for _, r in signal_log_df.head(10).iterrows():
                signal_rows += f"""
                <tr>
                    <td>{str(r['emitted_at'])[:16]}</td>
                    <td>{str(r['open_time'])[:16]}</td>
                    <td><span class="pill pill-green">{round(float(r['rel_score'])*100, 1)}%</span></td>
                    <td class="mono">${float(r['close']):.2f}</td>
                    <td><span class="pill pill-blue">{r['regime']}</span></td>
                </tr>"""

        run_rows = ""
        if len(run_log_df) > 0:
            for _, r in run_log_df.head(10).iterrows():
                status_html  = '<span class="pill pill-red">ABORTED</span>' if r["aborted"] \
                               else '<span class="pill pill-green">OK</span>'
                drift_html   = '<span class="pill pill-yellow">DRIFT</span>' if r["drift_detected"] else "—"
                retrain_html = '<span class="pill pill-purple">YES</span>' if r["retrained"] else "—"
                run_rows += f"""
                <tr>
                    <td class="mono">{str(r['run_start_utc'])[:19]}</td>
                    <td class="mono muted">{r['run_id'][:8]}</td>
                    <td>{status_html}</td>
                    <td class="mono">{r['new_rows_added']}</td>
                    <td>{drift_html}</td>
                    <td>{retrain_html}</td>
                    <td class="mono">{r['signals_emitted']}</td>
                </tr>"""

        ret_color  = "green" if total_return >= 0 else "red"
        wr_color   = "green" if win_rate >= 0.55 else "yellow"
        pf_color   = "green" if wf_pf >= 1.5 else "yellow"
        drft_color = "yellow" if drift_count > 0 else "green"

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # ───────────────────────────────── HTML PAGE ─────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="3600">
    <title>Claudex Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300..600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg:        #05070b;
            --surface:   #0b0f16;
            --surface-2: #111827;
            --surface-3: #111827;
            --border:    rgba(255,255,255,0.08);
            --border-2:  rgba(255,255,255,0.04);
            --text:      #e5e7eb;
            --muted:     #9ca3af;
            --faint:     #4b5563;
            --blue:      #60a5fa;
            --green:     #34d399;
            --red:       #f97373;
            --yellow:    #fbbf24;
            --purple:    #a78bfa;
            --font-body: 'Inter', -apple-system, system-ui, sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
            --radius:    7px;
            --radius-lg: 11px;
        }}
        [data-theme="light"] {{
            --bg:        #f5f5f4;
            --surface:   #ffffff;
            --surface-2: #f3f3f2;
            --surface-3: #e8e8e5;
            --border:    rgba(15,23,42,0.12);
            --border-2:  rgba(15,23,42,0.06);
            --text:      #111827;
            --muted:     #6b7280;
            --faint:     #d1d5db;
            --blue:      #2563eb;
            --green:     #059669;
            --red:       #dc2626;
            --yellow:    #d97706;
            --purple:    #7c3aed;
        }}
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: var(--font-body);
            background: radial-gradient(circle at top, #111827 0, var(--bg) 55%);
            color: var(--text);
            min-height: 100vh;
        }}
        .app {{
            max-width: 1180px;
            margin: 16px auto 32px auto;
            padding: 0 16px 32px 16px;
        }}
        .header {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 14px 18px;
            border-radius: var(--radius-lg);
            background: linear-gradient(120deg, rgba(96,165,250,0.13), rgba(16,185,129,0.05));
            border: 1px solid rgba(148,163,184,0.3);
            box-shadow: 0 18px 40px rgba(15,23,42,0.45);
            margin-bottom: 18px;
        }}
        .header-left h1 {{
            font-size: 1.15rem;
            letter-spacing: 0.04em;
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
        }}
        .model-badge {{
            font-size: 0.72rem;
            padding: 4px 8px;
            border-radius: 999px;
            background: rgba(15,23,42,0.75);
            border: 1px solid rgba(148,163,184,0.5);
        }}
        .status-label {{
            font-size: 0.72rem;
            color: var(--muted);
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .status-dot {{
            width: 7px; height: 7px; border-radius: 999px;
            background: var(--green);
            box-shadow: 0 0 8px rgba(34,197,94,0.9);
            animation: pulse 1.4s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 0.9; }}
            50% {{ transform: scale(1.6); opacity: 0.1; }}
            100% {{ transform: scale(1); opacity: 0.9; }}
        }}
        .theme-toggle {{
            background: rgba(15,23,42,0.7);
            border: 1px solid rgba(148,163,184,0.45);
            border-radius: 999px;
            width: 30px; height: 30px;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer;
            color: var(--muted);
            transition: background 150ms, color 150ms, transform 150ms;
        }}
        .theme-toggle:hover {{ background: rgba(15,23,42,1); color: var(--text); transform: translateY(-1px); }}

        .grid {{
            display: grid;
            grid-template-columns: 2fr 1.6fr;
            gap: 16px;
        }}
        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
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
            padding: 10px 12px;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,23,42,0.8));
            position: relative;
            overflow: hidden;
        }}
        .kpi::after {{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top left, rgba(96,165,250,0.16), transparent 55%);
            opacity: 0;
            transition: opacity 180ms ease-out;
        }}
        .kpi:hover::after {{ opacity: 1; }}
        .kpi-value {{
            font-size: 1.4rem;
            font-weight: 600;
        }}
        .kpi-label {{
            margin-top: 2px;
            font-size: 0.7rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
        }}

        .tab-row {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }}
        .tab-card {{
            position: relative;
            flex: 1 1 0;
            padding: 10px 11px;
            border-radius: var(--radius);
            border: 1px solid var(--border);
            background: var(--surface);
            cursor: default;
            transition: transform 140ms ease-out, box-shadow 140ms ease-out, border-color 140ms ease-out;
        }}
        .tab-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(15,23,42,0.55);
            border-color: rgba(96,165,250,0.8);
        }}
        .tab-title {{
            font-size: 0.78rem;
            font-weight: 500;
            margin-bottom: 2px;
        }}
        .tab-sub {{
            font-size: 0.72rem;
            color: var(--muted);
        }}
        .tab-tooltip {{
            position: absolute;
            z-index: 10;
            left: 50%;
            top: 100%;
            transform: translate(-50%, 6px);
            background: rgba(15,23,42,0.97);
            color: var(--text);
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 0.7rem;
            line-height: 1.4;
            width: 220px;
            border: 1px solid rgba(148,163,184,0.5);
            opacity: 0;
            pointer-events: none;
            transition: opacity 150ms ease-out, transform 150ms ease-out;
        }}
        .tab-card:hover .tab-tooltip {{
            opacity: 1;
            transform: translate(-50%, 0);
        }}

        .panel {{
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            background: var(--surface);
            padding: 12px 14px;
            margin-bottom: 16px;
        }}
        .panel-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
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
            display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
        }}
        .guidance-pill {{
            font-size: 0.8rem;
            padding: 5px 9px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.5);
            background: radial-gradient(circle at top, rgba(34,197,94,0.25), transparent 60%);
        }}
        .guidance-detail {{
            font-size: 0.75rem;
            color: var(--muted);
        }}
        .guidance-meta {{
            margin-top: 4px;
            font-size: 0.7rem;
            color: var(--muted);
        }}

        .chart-card {{
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            background: radial-gradient(circle at top, rgba(96,165,250,0.18), rgba(15,23,42,0.96));
            padding: 10px 12px 4px 12px;
            height: 230px;
        }}
        .chart-container {{
            position: relative;
            width: 100%;
            height: 170px;
        }}

        .table-card {{
            border-radius: var(--radius-lg);
            border: 1px solid var(--border);
            background: var(--surface);
            padding: 10px 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.76rem;
        }}
        thead tr {{
            background: rgba(15,23,42,0.9);
        }}
        th {{
            text-align: left;
            padding: 6px 6px;
            color: var(--muted);
            font-weight: 500;
        }}
        td {{
            padding: 5px 6px;
            border-top: 1px solid var(--border-2);
        }}
        tbody tr:hover {{
            background: rgba(15,23,42,0.7);
        }}
        .mono {{ font-family: var(--font-mono); font-feature-settings: "tnum" 1; }}
        .muted {{ color: var(--muted); }}

        .pill {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 2px 7px;
            font-size: 0.7rem;
        }}
        .pill-green  {{ background: rgba(22,163,74,0.15);  color: #4ade80; }}
        .pill-blue   {{ background: rgba(37,99,235,0.2);   color: #93c5fd; }}
        .pill-gray   {{ background: rgba(148,163,184,0.18); color: var(--muted); }}
        .pill-yellow {{ background: rgba(234,179,8,0.18);   color: #facc15; }}
        .pill-red    {{ background: rgba(248,113,113,0.18); color: #fecaca; }}
        .pill-purple {{ background: rgba(129,140,248,0.18); color: #c7d2fe; }}
    </style>
</head>
<body>
<div class="app">

    <header class="header">
        <div class="header-left">
            <h1>⚡ Claudex · ETHUSD 1H</h1>
            <p>Self-improving compression specialist · band ge_0.78 · Run {run_id}</p>
        </div>
        <div class="header-right">
            <span class="model-badge">{active_model}</span>
            <button class="theme-toggle" data-theme-toggle aria-label="Switch to light mode">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2">
                    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
            </button>
            <span class="status-label">
                <span class="status-dot"></span>
                Live · {generated_at}
            </span>
        </div>
    </header>

    <!-- KPI + Tabs row -->
    <section style="margin-bottom:14px;">
        <div class="kpi-row">
            <div class="kpi">
                <div class="kpi-value" style="color:var(--{ret_color});">
                    {total_return:+.2%}
                </div>
                <div class="kpi-label">Total Return</div>
            </div>
            <div class="kpi">
                <div class="kpi-value" style="color:var(--{wr_color});">
                    {win_rate:.1%}
                </div>
                <div class="kpi-label">Win Rate</div>
            </div>
            <div class="kpi">
                <div class="kpi-value">{total_trades}</div>
                <div class="kpi-label">Live Trades</div>
            </div>
            <div class="kpi">
                <div class="kpi-value">{signals_total}</div>
                <div class="kpi-label">Signals Emitted</div>
            </div>
            <div class="kpi">
                <div class="kpi-value" style="color:var(--{pf_color});">
                    {wf_pf:.2f}
                </div>
                <div class="kpi-label">WF Profit Factor</div>
            </div>
            <div class="kpi">
                <div class="kpi-value" style="color:var(--{drft_color});">
                    {drift_count}
                </div>
                <div class="kpi-label">Drift Events</div>
            </div>
        </div>

        <!-- Tab-cards with hover tooltips -->
        <div class="tab-row">
            <div class="tab-card">
                <div class="tab-title">Model Health</div>
                <div class="tab-sub">PF {wf_pf:.2f} · OOS {wf_return:+.2%}</div>
                <div class="tab-tooltip">
                    Walk-forward validated edge for the currently active specialist
                    at band ge_0.78. Profit factor and OOS return drive whether
                    retrains are promoted.
                </div>
            </div>
            <div class="tab-card">
                <div class="tab-title">Drift Monitor</div>
                <div class="tab-sub">{drift_count} recent events</div>
                <div class="tab-tooltip">
                    Drift checker compares recent paper-trade PF and win rate
                    against the baseline band result. If PF falls below 90% of
                    baseline, a retrain is queued.
                </div>
            </div>
            <div class="tab-card">
                <div class="tab-title">Signal Quality</div>
                <div class="tab-sub">{signals_total} live signals</div>
                <div class="tab-tooltip">
                    High-confidence compression signals filtered by calibrated
                    rel_score ≥ 0.78. Each signal emits email + toast once.
                </div>
            </div>
        </div>
    </section>

    <div class="grid">
        <!-- LEFT: Guidance + Equity -->
        <div>
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Trade Guidance (Model View)</span>
                    <span class="panel-tag">Not financial advice — model signal only</span>
                </div>
                <div class="guidance-main">
                    <span class="guidance-pill mono">{suggestion}</span>
                    <div class="guidance-detail">{suggestion_detail}</div>
                </div>
                <div class="guidance-meta">
                    Option type: <span class="mono">{option_type}</span> · Side:
                    <span class="mono">{side}</span> · Target date:
                    <span class="mono">{target_date}</span>
                </div>
            </div>

            <div class="chart-card">
                <div class="panel-header" style="margin-bottom:4px;">
                    <span class="panel-title">Equity Curve · Paper Account</span>
                    <span class="panel-tag">Base 100,000 · {last_run}</span>
                </div>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>

            <div class="table-card">
                <div class="panel-header">
                    <span class="panel-title">Recent Trades (Last 20)</span>
                    <span class="panel-tag">Live paper_trades</span>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Entry</th><th>Entry $</th><th>Exit $</th>
                            <th>Conf</th><th>Exit</th>
                            <th>Return</th><th>PnL</th><th>Hold</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trade_rows if trade_rows else '<tr><td colspan="8" class="muted" style="text-align:center;padding:10px;">No trades yet</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- RIGHT: ETH chart + signals + runs -->
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
                            <th>Emitted</th><th>Bar Open</th><th>Rel%</th><th>Close</th><th>Regime</th>
                        </tr>
                    </thead>
                    <tbody>
                        {signal_rows if signal_rows else '<tr><td colspan="5" class="muted" style="text-align:center;padding:10px;">No signals yet</td></tr>'}
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
                            <th>Started (UTC)</th><th>Run ID</th><th>Status</th>
                            <th>New</th><th>Drift</th><th>Retrain</th><th>Signals</th>
                        </tr>
                    </thead>
                    <tbody>
                        {run_rows if run_rows else '<tr><td colspan="7" class="muted" style="text-align:center;padding:10px;">No runs logged yet</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</div>

<script>
(function() {{
    // ── Theme toggle ──────────────────────────────────────────────────────
    const root   = document.documentElement;
    const toggle = document.querySelector('[data-theme-toggle]');
    let theme    = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';

    function applyTheme(t) {{
        theme = t;
        root.setAttribute('data-theme', t);
        if (toggle) {{
            toggle.setAttribute('aria-label',
                'Switch to ' + (t === 'dark' ? 'light' : 'dark') + ' mode');
            toggle.innerHTML = t === 'dark'
                ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
                : '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
        }}
    }}
    applyTheme(theme);
    toggle && toggle.addEventListener('click', () =>
        applyTheme(theme === 'dark' ? 'light' : 'dark')
    );

    // ── Equity chart ──────────────────────────────────────────────────────
    const eqLabels = {equity_labels_js};
    const eqValues = {equity_values_js};
    if (eqLabels.length && eqValues.length) {{
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: eqLabels,
                datasets: [{{
                    label: 'Equity Return %',
                    data: eqValues,
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96,165,250,0.18)',
                    borderWidth: 2,
                    pointRadius: 0,
                    pointHitRadius: 6,
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
                        mode: 'index',
                        intersect: false,
                    }},
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#9ca3af', maxTicksLimit: 7 }},
                        grid: {{ color: 'rgba(31,41,55,0.6)' }},
                    }},
                    y: {{
                        ticks: {{ color: '#9ca3af', callback: v => v + '%' }},
                        grid: {{ color: 'rgba(31,41,55,0.6)' }},
                    }},
                }},
            }}
        }});
    }}

    // ── ETH nowcast + hypothetical chart ──────────────────────────────────
    const pLabels = {price_labels_js};
    const pNow    = {price_now_js};
    const pFut    = {price_future_js};
    if (pLabels.length && pNow.length) {{
        new Chart(document.getElementById('priceChart'), {{
            type: 'line',
            data: {{
                labels: pLabels.concat(pFut.length ? ['+1h','+2h'] : []),
                datasets: [
                    {{
                        label: 'ETHUSD close',
                        data: pNow,
                        borderColor: '#34d399',
                        backgroundColor: 'rgba(16,185,129,0.16)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHitRadius: 6,
                        fill: true,
                        tension: 0.3,
                    }},
                    pFut.length ? {{
                        label: 'Hypothetical path',
                        data: pFut,
                        borderColor: '#fbbf24',
                        borderWidth: 1.5,
                        borderDash: [5,4],
                        pointRadius: 0,
                        fill: false,
                        tension: 0.4,
                    }} : null,
                ].filter(Boolean)
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2)
                        }},
                    }},
                }},
                scales: {{
                    x: {{
                        ticks: {{ color: '#9ca3af', maxTicksLimit: 7 }},
                        grid: {{ color: 'rgba(31,41,55,0.6)' }},
                    }},
                    y: {{
                        ticks: {{ color: '#9ca3af' }},
                        grid: {{ color: 'rgba(31,41,55,0.6)' }},
                    }},
                }},
            }}
        }});
    }}
}})();
</script>

</body>
</html>"""

        output_path.write_text(html, encoding="utf-8")

        return {
            "status":        "success",
            "dashboard_path": str(output_path),
            "total_trades":  total_trades,
            "win_rate":      round(win_rate, 4),
            "signals_total": signals_total,
            "generated_at":  generated_at,
        }

    except Exception:
        con.close()
        return {
            "status": "failed",
            "error":  traceback.format_exc(),
            "dashboard_path": "",
        }