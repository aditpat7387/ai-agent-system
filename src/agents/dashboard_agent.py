# =============================================================================
# dashboard_agent.py
# Wraps generate_model_dashboard.py as a typed agent tool
# Reads entirely from DuckDB — no CSV file dependencies
# Called by orchestrator as the final reporting step
# =============================================================================

import sys
import traceback
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def run_dashboard_agent(cfg: dict, context: dict) -> dict:
    """
    Typed tool contract:
    INPUT  cfg     : full agent_config.yaml as dict
    INPUT  context : shared orchestrator context dict
    OUTPUT dict    : {status, dashboard_path, error}
    """
    dash_cfg = cfg["dashboard_agent"]
    paths    = cfg["paths"]
    tables   = cfg["tables"]

    db_path        = PROJECT_ROOT / paths["db_path"]
    output_path    = PROJECT_ROOT / dash_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(db_path))

    try:
        # ── 1. Load data from DuckDB ─────────────────────────────────────────
        run_id = context.get("run_id", "unknown")

        # Recent paper trades
        try:
            trades_df = con.execute(f"""
                SELECT * FROM paper_trade_agent_log
                ORDER BY entry_time DESC
                LIMIT {dash_cfg['lookback_trades']}
            """).df()
        except Exception:
            trades_df = pd.DataFrame()

        # Recent run log
        try:
            run_log_df = con.execute("""
                SELECT * FROM run_log
                ORDER BY run_start_utc DESC
                LIMIT 20
            """).df()
        except Exception:
            run_log_df = pd.DataFrame()

        # Recent signals
        try:
            signal_log_df = con.execute("""
                SELECT * FROM signal_log
                ORDER BY signal_time_utc DESC
                LIMIT 20
            """).df()
        except Exception:
            signal_log_df = pd.DataFrame()

        # Drift log
        try:
            drift_log_df = con.execute("""
                SELECT * FROM drift_log
                ORDER BY check_time_utc DESC
                LIMIT 10
            """).df()
        except Exception:
            drift_log_df = pd.DataFrame()

        # Model registry
        try:
            model_reg_df = con.execute("""
                SELECT * FROM model_registry
                ORDER BY promoted_at_utc DESC
                LIMIT 5
            """).df()
        except Exception:
            model_reg_df = pd.DataFrame()

        # Band walk-forward summary
        try:
            wf_summary_df = con.execute(f"""
                SELECT * FROM {tables['compression_selected']}
            """).df()
        except Exception:
            wf_summary_df = pd.DataFrame()

        con.close()

        # ── 2. Compute KPIs ──────────────────────────────────────────────────
        total_trades  = len(trades_df)
        win_rate      = float((trades_df["net_return"] > 0).mean()) if total_trades > 0 else 0.0
        total_return  = float(
            trades_df["account_equity"].iloc[0] / 100000.0 - 1.0
        ) if total_trades > 0 else 0.0
        signals_total = len(signal_log_df)
        last_run      = run_log_df["run_start_utc"].iloc[0] if len(run_log_df) > 0 else "N/A"
        drift_count   = int(drift_log_df["drift_detected"].sum()) if len(drift_log_df) > 0 else 0

        active_model = "compression_specialist_v7.joblib"
        if len(model_reg_df) > 0:
            active_row = model_reg_df[model_reg_df["is_active"] == True]
            if not active_row.empty:
                active_model = active_row["model_version"].iloc[0]

        wf_pf      = float(wf_summary_df["profit_factor"].iloc[0])    if len(wf_summary_df) > 0 else 0.0
        wf_return  = float(wf_summary_df["oos_total_net_return"].iloc[0]) if len(wf_summary_df) > 0 else 0.0
        wf_winrate = float(wf_summary_df["oos_win_rate"].iloc[0])      if len(wf_summary_df) > 0 else 0.0

        # ── 3. Build equity curve rows ───────────────────────────────────────
        equity_rows = ""
        if total_trades > 0:
            for _, r in trades_df.iloc[::-1].iterrows():
                eq_pct = round((r["account_equity"] / 100000.0 - 1.0) * 100, 2)
                equity_rows += f"['{str(r['entry_time'])[:16]}', {eq_pct}],"

        # ── 4. Build trade table rows ────────────────────────────────────────
        trade_rows = ""
        if total_trades > 0:
            for _, r in trades_df.head(20).iterrows():
                icon    = "✅" if r["net_return"] > 0 else "❌"
                ret_pct = round(r["net_return"] * 100, 3)
                trade_rows += f"""
                <tr>
                    <td>{str(r['entry_time'])[:16]}</td>
                    <td>{round(r['entry_price'], 2)}</td>
                    <td>{round(r['exit_price'], 2)}</td>
                    <td>{round(r['specialist_pred_proba'], 3)}</td>
                    <td>{r['exit_reason']}</td>
                    <td>{icon} {ret_pct}%</td>
                    <td>${round(r['pnl_dollars'], 2)}</td>
                </tr>"""

        # ── 5. Build run log rows ────────────────────────────────────────────
        run_rows = ""
        if len(run_log_df) > 0:
            for _, r in run_log_df.head(10).iterrows():
                status_icon = "🔴 ABORTED" if r["aborted"] else "🟢 OK"
                run_rows += f"""
                <tr>
                    <td>{str(r['run_start_utc'])[:19]}</td>
                    <td>{r['run_id']}</td>
                    <td>{status_icon}</td>
                    <td>{r['new_rows_added']}</td>
                    <td>{'⚠️ YES' if r['drift_detected'] else 'No'}</td>
                    <td>{'🔄 YES' if r['retrained'] else 'No'}</td>
                    <td>{r['signals_emitted']}</td>
                </tr>"""

        # ── 6. Render HTML ───────────────────────────────────────────────────
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="3600">
    <title>Claudex Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }}
        .header {{ background: #161b22; padding: 20px 30px; border-bottom: 1px solid #30363d; }}
        .header h1 {{ color: #58a6ff; font-size: 1.6rem; }}
        .header p {{ color: #8b949e; font-size: 0.85rem; margin-top: 4px; }}
        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                     gap: 16px; padding: 24px 30px; }}
        .kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 16px; text-align: center; }}
        .kpi .value {{ font-size: 1.8rem; font-weight: 700; color: #58a6ff; }}
        .kpi .label {{ font-size: 0.75rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; }}
        .kpi.green .value {{ color: #3fb950; }}
        .kpi.red .value   {{ color: #f85149; }}
        .kpi.yellow .value {{ color: #d29922; }}
        .section {{ padding: 0 30px 30px; }}
        .section h2 {{ color: #58a6ff; font-size: 1rem; margin-bottom: 12px;
                       padding-bottom: 6px; border-bottom: 1px solid #30363d; }}
        .chart-box {{ background: #161b22; border: 1px solid #30363d;
                      border-radius: 8px; padding: 16px; height: 260px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
        th {{ background: #21262d; color: #8b949e; padding: 8px 10px;
              text-align: left; font-weight: 500; }}
        td {{ padding: 7px 10px; border-bottom: 1px solid #21262d; }}
        tr:hover {{ background: #161b22; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 0.75rem; font-weight: 600; }}
        .badge.green {{ background: #1a3a2a; color: #3fb950; }}
        .badge.blue  {{ background: #1a2a3a; color: #58a6ff; }}
        .wf-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
        .wf-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }}
        .wf-card .wf-val {{ font-size: 1.3rem; font-weight: 700; color: #3fb950; }}
        .wf-card .wf-lbl {{ font-size: 0.72rem; color: #8b949e; margin-top: 2px; }}
    </style>
</head>
<body>

<div class="header">
    <h1>⚡ Claudex — Live Trading Dashboard</h1>
    <p>Generated: {generated_at} &nbsp;|&nbsp; Run ID: {run_id} &nbsp;|&nbsp;
       Active Model: {active_model} &nbsp;|&nbsp; Band: ge_0.78</p>
</div>

<!-- KPI Cards -->
<div class="kpi-grid">
    <div class="kpi {'green' if total_return >= 0 else 'red'}">
        <div class="value">{'+' if total_return >= 0 else ''}{round(total_return * 100, 2)}%</div>
        <div class="label">Total Return</div>
    </div>
    <div class="kpi {'green' if win_rate >= 0.55 else 'yellow'}">
        <div class="value">{round(win_rate * 100, 1)}%</div>
        <div class="label">Win Rate</div>
    </div>
    <div class="kpi">
        <div class="value">{total_trades}</div>
        <div class="label">Total Trades</div>
    </div>
    <div class="kpi blue">
        <div class="value">{signals_total}</div>
        <div class="label">Signals Emitted</div>
    </div>
    <div class="kpi {'green' if wf_pf >= 1.5 else 'yellow'}">
        <div class="value">{round(wf_pf, 2)}</div>
        <div class="label">WF Profit Factor</div>
    </div>
    <div class="kpi {'yellow' if drift_count > 0 else 'green'}">
        <div class="value">{drift_count}</div>
        <div class="label">Drift Events</div>
    </div>
    <div class="kpi">
        <div class="value">{str(last_run)[:16]}</div>
        <div class="label">Last Run</div>
    </div>
</div>

<!-- Walk-Forward Validated Edge -->
<div class="section">
    <h2>Walk-Forward Validated Edge (ge_0.78 Band)</h2>
    <div class="wf-grid">
        <div class="wf-card">
            <div class="wf-val">{round(wf_return * 100, 2)}%</div>
            <div class="wf-lbl">OOS Total Return</div>
        </div>
        <div class="wf-card">
            <div class="wf-val">{round(wf_winrate * 100, 1)}%</div>
            <div class="wf-lbl">OOS Win Rate</div>
        </div>
        <div class="wf-card">
            <div class="wf-val">{round(wf_pf, 3)}</div>
            <div class="wf-lbl">OOS Profit Factor</div>
        </div>
    </div>
</div>

<!-- Equity Curve -->
<div class="section">
    <h2>Equity Curve</h2>
    <div class="chart-box">
        <canvas id="equityChart"></canvas>
    </div>
</div>

<!-- Recent Trades -->
<div class="section">
    <h2>Recent Trades (Last 20)</h2>
    <table>
        <thead>
            <tr>
                <th>Entry Time</th><th>Entry $</th><th>Exit $</th>
                <th>Confidence</th><th>Exit Reason</th>
                <th>Net Return</th><th>PnL $</th>
            </tr>
        </thead>
        <tbody>{trade_rows if trade_rows else '<tr><td colspan="7" style="text-align:center;color:#8b949e">No trades yet</td></tr>'}</tbody>
    </table>
</div>

<!-- Run Log -->
<div class="section">
    <h2>Orchestrator Run Log (Last 10)</h2>
    <table>
        <thead>
            <tr>
                <th>Started (UTC)</th><th>Run ID</th><th>Status</th>
                <th>New Rows</th><th>Drift</th><th>Retrained</th><th>Signals</th>
            </tr>
        </thead>
        <tbody>{run_rows if run_rows else '<tr><td colspan="7" style="text-align:center;color:#8b949e">No runs logged yet</td></tr>'}</tbody>
    </table>
</div>

<script>
const labels = [{equity_rows}].map(r => r[0]);
const data   = [{equity_rows}].map(r => r[1]);
new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
        labels: labels,
        datasets: [{{
            label: 'Equity Return %',
            data: data,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.08)',
            borderWidth: 2,
            pointRadius: 2,
            fill: true,
            tension: 0.3,
        }}]
    }},
    options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 8 }},
                  grid: {{ color: '#21262d' }} }},
            y: {{ ticks: {{ color: '#8b949e' }},
                  grid: {{ color: '#21262d' }} }}
        }}
    }}
}});
</script>

</body>
</html>"""

        output_path.write_text(html, encoding="utf-8")

        return {
            "status":         "success",
            "dashboard_path": str(output_path),
            "total_trades":   total_trades,
            "win_rate":       round(win_rate, 4),
            "signals_total":  signals_total,
            "generated_at":   generated_at,
        }

    except Exception as e:
        con.close()
        return {
            "status": "failed",
            "error":  traceback.format_exc(),
            "dashboard_path": "",
        }
