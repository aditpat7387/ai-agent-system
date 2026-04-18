from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def load_summary(path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    return rows[0] if rows else {}


def load_threshold_rows(path: Path) -> List[Dict[str, Any]]:
    return read_csv_rows(path)


def load_regime_actual_rows(path: Path) -> List[Dict[str, Any]]:
    return read_csv_rows(path)


def load_trade_rows(path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    rows = read_csv_rows(path)
    return rows[:limit]


def build_regime_insight(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    compression_total = 0
    compression_pos = 0
    compression_neg = 0
    all_rows = 0

    for r in rows:
        regime = (r.get("regime_label") or "").strip()
        event_label = to_int(r.get("event_label"), 0)
        cnt = to_int(r.get("rows"), 0)
        all_rows += cnt
        if regime == "compression":
            compression_total += cnt
            if event_label == 1:
                compression_pos += cnt
            else:
                compression_neg += cnt

    compression_positive_rate = (compression_pos / compression_total) if compression_total else 0.0
    compression_share = (compression_total / all_rows) if all_rows else 0.0

    return {
        "compression_total": compression_total,
        "compression_pos": compression_pos,
        "compression_neg": compression_neg,
        "compression_positive_rate": compression_positive_rate,
        "compression_share": compression_share,
        "all_rows": all_rows,
    }


def threshold_comment(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "Threshold sweep file not found."

    trades = {to_int(r.get("trades"), 0) for r in rows}
    returns = {round(to_float(r.get("total_net_return"), 0.0), 10) for r in rows}
    drawdowns = {round(to_float(r.get("max_drawdown"), 0.0), 10) for r in rows}

    if len(trades) == 1 and len(returns) == 1 and len(drawdowns) == 1:
        return (
            "Threshold sweep looks flat across the tested band, which usually means the current trade set is unchanged "
            "and threshold tuning alone is unlikely to improve the strategy materially."
        )

    best = max(rows, key=lambda r: to_float(r.get("total_net_return"), -999))
    return (
        f"Best tested threshold by total return is {best.get('threshold', 'n/a')} "
        f"with total net return {pct(to_float(best.get('total_net_return', 0.0)))}."
    )


def render_table(rows: List[Dict[str, Any]], columns: List[str], title_map: Dict[str, str] | None = None) -> str:
    if not rows:
        return '<p class="empty">No data available.</p>'

    title_map = title_map or {}
    thead = "".join(f"<th>{html.escape(title_map.get(c, c))}</th>" for c in columns)
    body_parts = []

    for row in rows:
        tds = []
        for c in columns:
            val = row.get(c, "")
            if c in {"win_rate", "avg_net_return", "total_net_return", "max_drawdown"}:
                try:
                    val = pct(float(val))
                except Exception:
                    pass
            elif c in {"entry_price", "exit_price", "account_equity", "pnl_dollars", "final_equity"}:
                try:
                    val = num(float(val))
                except Exception:
                    pass
            tds.append(f"<td>{html.escape(str(val))}</td>")
        body_parts.append(f"<tr>{''.join(tds)}</tr>")

    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>{''.join(body_parts)}</tbody>
      </table>
    </div>
    """


def build_html(
    summary: Dict[str, Any],
    threshold_rows: List[Dict[str, Any]],
    regime_rows: List[Dict[str, Any]],
    trade_rows: List[Dict[str, Any]],
    source_dir: Path,
) -> str:
    insight = build_regime_insight(regime_rows)
    comment = threshold_comment(threshold_rows)

    strategy_name = summary.get("strategy_name", "compression_specialist_v7")
    threshold = to_float(summary.get("threshold"), 0.0)
    trades = to_int(summary.get("trades"), 0)
    win_rate = to_float(summary.get("win_rate"), 0.0)
    total_ret = to_float(summary.get("total_net_return"), 0.0)
    max_dd = to_float(summary.get("max_drawdown"), 0.0)
    final_equity = to_float(summary.get("final_equity"), 0.0)
    avg_pred = to_float(summary.get("avg_specialist_pred_proba"), 0.0)
    rejections = to_int(summary.get("rejections"), 0)
    avg_hold = to_float(summary.get("avg_bars_held"), 0.0)
    tp_count = to_int(summary.get("take_profit_count"), 0)
    sl_count = to_int(summary.get("stop_loss_count"), 0)
    time_exit_count = to_int(summary.get("time_exit_count"), 0)

    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_manifest = {
        "summary": str(source_dir / "compression_specialist_summary_v7.csv"),
        "threshold": str(source_dir / "compression_specialist_threshold_sweep_v7.csv"),
        "regime": str(source_dir / "regime_actual_summary_v7.csv"),
        "trades": str(source_dir / "compression_specialist_trades_v7.csv"),
    }

    threshold_table = render_table(
        threshold_rows,
        ["threshold", "trades", "win_rate", "total_net_return", "max_drawdown", "rejections"],
        {
            "threshold": "Threshold",
            "trades": "Trades",
            "win_rate": "Win Rate",
            "total_net_return": "Total Return",
            "max_drawdown": "Max Drawdown",
            "rejections": "Rejections",
        },
    )

    trade_table = render_table(
        trade_rows,
        ["entry_time", "exit_time", "entry_price", "exit_price", "specialist_pred_proba", "actual_label", "exit_reason", "pnl_dollars", "account_equity"],
        {
            "entry_time": "Entry Time",
            "exit_time": "Exit Time",
            "entry_price": "Entry Px",
            "exit_price": "Exit Px",
            "specialist_pred_proba": "Pred Proba",
            "actual_label": "Actual",
            "exit_reason": "Exit",
            "pnl_dollars": "PnL $",
            "account_equity": "Equity",
        },
    )

    regime_table = render_table(
        regime_rows,
        ["regime_label", "event_label", "rows"],
        {
            "regime_label": "Regime",
            "event_label": "Label",
            "rows": "Rows",
        },
    )

    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>model-insights-dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root, [data-theme="light"] {{
      --font-display: 'Instrument Serif', Georgia, serif;
      --font-body: 'Inter', system-ui, sans-serif;
      --text-xs: clamp(0.75rem, 0.7rem + 0.25vw, 0.875rem);
      --text-sm: clamp(0.875rem, 0.8rem + 0.35vw, 1rem);
      --text-base: clamp(1rem, 0.95rem + 0.25vw, 1.125rem);
      --text-lg: clamp(1.125rem, 1rem + 0.75vw, 1.5rem);
      --text-xl: clamp(1.5rem, 1.2rem + 1.25vw, 2.25rem);
      --text-2xl: clamp(2rem, 1.2rem + 2.5vw, 3.5rem);
      --bg: #f7f6f2;
      --surface: #fbfbf9;
      --surface-2: #f3f0ec;
      --text: #28251d;
      --muted: #6c6a64;
      --border: #d4d1ca;
      --primary: #01696f;
      --primary-soft: #dceceb;
      --success: #437a22;
      --warning: #964219;
      --danger: #a12c7b;
      --shadow: 0 10px 30px rgba(20,20,20,.08);
      --radius: 18px;
    }}
    [data-theme="dark"] {{
      --bg: #171614;
      --surface: #1c1b19;
      --surface-2: #23211f;
      --text: #e3e0da;
      --muted: #a39f97;
      --border: #393836;
      --primary: #4f98a3;
      --primary-soft: #223437;
      --success: #7fb15c;
      --warning: #d18b5e;
      --danger: #d163a7;
      --shadow: 0 10px 30px rgba(0,0,0,.28);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: var(--font-body); background: var(--bg); color: var(--text); line-height: 1.6; }}
    .container {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; }}
    .topbar {{ position: sticky; top: 0; z-index: 20; backdrop-filter: blur(14px); background: color-mix(in srgb, var(--bg) 86%, transparent); border-bottom: 1px solid var(--border); }}
    .topbar-inner {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 0; }}
    .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 700; }}
    .logo {{ width: 34px; height: 34px; border-radius: 10px; background: linear-gradient(135deg, var(--primary), color-mix(in srgb, var(--primary) 55%, black)); display: grid; place-items: center; color: white; box-shadow: var(--shadow); }}
    .toggle {{ min-width: 44px; min-height: 44px; border-radius: 999px; border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer; }}
    .hero {{ padding: 44px 0 20px; }}
    .eyebrow {{ font-size: var(--text-xs); text-transform: uppercase; letter-spacing: .16em; color: var(--muted); margin-bottom: 10px; }}
    h1 {{ font-family: var(--font-display); font-size: var(--text-2xl); line-height: 1; margin: 0 0 14px; max-width: 12ch; }}
    .lede {{ font-size: var(--text-lg); color: var(--muted); max-width: 70ch; margin: 0; }}
    .hero-grid {{ display: grid; grid-template-columns: 1.2fr .8fr; gap: 20px; align-items: end; }}
    .hero-card {{ background: radial-gradient(circle at top left, var(--primary-soft), transparent 55%), var(--surface); border: 1px solid var(--border); border-radius: 28px; padding: 22px; box-shadow: var(--shadow); }}
    .section {{ padding: 22px 0; }}
    .section-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }}
    .section-title {{ font-size: var(--text-xl); font-family: var(--font-display); margin: 0; }}
    .section-copy {{ margin: 0; color: var(--muted); max-width: 70ch; }}
    .grid {{ display: grid; gap: 16px; }}
    .kpi-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; box-shadow: var(--shadow); }}
    .kpi-label {{ font-size: var(--text-xs); text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin-bottom: 8px; }}
    .kpi-value {{ font-size: clamp(1.6rem, 1.2rem + 1vw, 2.3rem); font-weight: 800; line-height: 1.05; }}
    .kpi-sub {{ margin-top: 8px; color: var(--muted); font-size: var(--text-sm); }}
    .accent {{ color: var(--primary); }}
    .good {{ color: var(--success); }}
    .bad {{ color: var(--danger); }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .story p {{ margin: 0 0 12px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .pill {{ padding: 8px 12px; background: var(--surface-2); border: 1px solid var(--border); border-radius: 999px; font-size: var(--text-sm); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--border); border-radius: 16px; background: var(--surface); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--border); font-size: var(--text-sm); vertical-align: top; }}
    th {{ background: var(--surface-2); }}
    .footer {{ padding: 24px 0 60px; color: var(--muted); font-size: var(--text-sm); }}
    pre.meta {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 12px; color: var(--muted); }}
    .empty {{ color: var(--muted); }}
    @media (max-width: 920px) {{
      .hero-grid, .two-col, .kpi-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 640px) {{
      .hero-grid, .two-col, .kpi-grid {{ grid-template-columns: 1fr; }}
      h1 {{ max-width: 100%; }}
      .container {{ width: min(1180px, calc(100% - 20px)); }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="container topbar-inner">
      <div class="brand">
        <div class="logo">M</div>
        <div>
          <div>Model Insights Dashboard</div>
          <div style="font-size:12px;color:var(--muted);">{html.escape(strategy_name)}</div>
        </div>
      </div>
      <button class="toggle" data-theme-toggle aria-label="Toggle theme">◐</button>
    </div>
  </header>

  <main>
    <section class="hero">
      <div class="container hero-grid">
        <div>
          <div class="eyebrow">Current model • local html dashboard</div>
          <h1>Compression specialist insights from your latest outputs.</h1>
          <p class="lede">A text-first dashboard that converts your current CSV artifacts into a clean operational view across regime mix, backtest quality, threshold behavior, and recent trade outcomes.</p>
        </div>
        <div class="hero-card">
          <div class="kpi-label">Model stance</div>
          <p>The strategy is active but not yet robust. It is trading the compression regime selectively, however the current backtest still shows a small negative return and meaningful drawdown.</p>
          <p style="color:var(--muted);font-size:14px;">Generated at {html.escape(generation_time)}</p>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Core KPIs</h2>
            <p class="section-copy">Fast health checks for the current compression specialist snapshot.</p>
          </div>
        </div>
        <div class="grid kpi-grid">
          <article class="card"><div class="kpi-label">Threshold</div><div class="kpi-value accent">{threshold:.2f}</div><div class="kpi-sub">Current backtest threshold</div></article>
          <article class="card"><div class="kpi-label">Trades</div><div class="kpi-value">{trades}</div><div class="kpi-sub">Executed trades</div></article>
          <article class="card"><div class="kpi-label">Win rate</div><div class="kpi-value">{pct(win_rate)}</div><div class="kpi-sub">Across executed trades</div></article>
          <article class="card"><div class="kpi-label">Final equity</div><div class="kpi-value">{num(final_equity, 0)}</div><div class="kpi-sub">Ending equity</div></article>
          <article class="card"><div class="kpi-label">Total return</div><div class="kpi-value {'good' if total_ret > 0 else 'bad'}">{pct(total_ret)}</div><div class="kpi-sub">Net strategy return</div></article>
          <article class="card"><div class="kpi-label">Max drawdown</div><div class="kpi-value bad">{pct(max_dd)}</div><div class="kpi-sub">Peak-to-trough</div></article>
          <article class="card"><div class="kpi-label">Avg pred proba</div><div class="kpi-value">{avg_pred:.3f}</div><div class="kpi-sub">Mean prediction probability</div></article>
          <article class="card"><div class="kpi-label">Rejections</div><div class="kpi-value">{rejections}</div><div class="kpi-sub">Filtered rows</div></article>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="container two-col">
        <article class="card story">
          <h2 class="section-title">Model reading</h2>
          <p>Compression has <strong>{insight['compression_total']}</strong> rows, made up of <strong>{insight['compression_neg']}</strong> negative labels and <strong>{insight['compression_pos']}</strong> positive labels.</p>
          <p>The positive event rate inside compression is <strong>{pct(insight['compression_positive_rate'])}</strong>, and compression contributes <strong>{pct(insight['compression_share'])}</strong> of all rows in the regime summary.</p>
          <p>Average holding period is <strong>{avg_hold:.2f}</strong> bars, with exit mix of <strong>{tp_count}</strong> take-profit, <strong>{sl_count}</strong> stop-loss, and <strong>{time_exit_count}</strong> time exits.</p>
          <div class="pill-row">
            <span class="pill">Compression rows: {insight['compression_total']}</span>
            <span class="pill">Positive labels: {insight['compression_pos']}</span>
            <span class="pill">Negative labels: {insight['compression_neg']}</span>
          </div>
        </article>
        <article class="card story">
          <h2 class="section-title">Recommendation</h2>
          <p><strong>Preferred next step:</strong> keep the dashboard, but prioritize model discrimination and walk-forward robustness before further threshold tuning.</p>
          <p>{html.escape(comment)}</p>
          <p>If the threshold sweep remains flat, focus on feature refinement, labeling review, calibration validation, and out-of-sample testing rather than micro-adjusting the cutoff.</p>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Threshold sweep</h2>
            <p class="section-copy">Compact view of the tested threshold band.</p>
          </div>
        </div>
        {threshold_table}
      </div>
    </section>

    <section class="section">
      <div class="container two-col">
        <article>
          <div class="section-head">
            <div>
              <h2 class="section-title">Regime summary</h2>
              <p class="section-copy">Actual label counts by regime from the diagnostics file.</p>
            </div>
          </div>
          {regime_table}
        </article>
        <article class="card story">
          <h2 class="section-title">Operational notes</h2>
          <p>This page is intentionally text based, so it fits well into a scheduled workflow and remains easy to regenerate after every run.</p>
          <p>You can point the script to any output folder, and it writes a self-contained HTML dashboard there using the latest CSV files from the source directory.</p>
          <pre class="meta">{html.escape(json.dumps(file_manifest, indent=2))}</pre>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="container">
        <div class="section-head">
          <div>
            <h2 class="section-title">Recent trades</h2>
            <p class="section-copy">Sample rows from the latest trade log.</p>
          </div>
        </div>
        {trade_table}
      </div>
    </section>
  </main>

  <footer class="footer">
    <div class="container">Generated by <code>generate_model_dashboard.py</code>.</div>
  </footer>

  <script>
    (function () {{
      const btn = document.querySelector('[data-theme-toggle]');
      const root = document.documentElement;
      let mode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      root.setAttribute('data-theme', mode);
      btn.addEventListener('click', function () {{
        mode = mode === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', mode);
      }});
    }})();
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a text-based model insights dashboard HTML from CSV artifacts.")
    parser.add_argument("--source-dir", required=True, help="Directory containing the CSV files.")
    parser.add_argument("--output-path", required=True, help="Full output HTML path to generate.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = load_summary(source_dir / "compression_specialist_summary_v7.csv")
    threshold_rows = load_threshold_rows(source_dir / "compression_specialist_threshold_sweep_v7.csv")
    regime_rows = load_regime_actual_rows(source_dir / "regime_actual_summary_v7.csv")
    trade_rows = load_trade_rows(source_dir / "compression_specialist_trades_v7.csv", limit=12)

    html_text = build_html(summary, threshold_rows, regime_rows, trade_rows, source_dir)
    output_path.write_text(html_text, encoding="utf-8")

    print(f"[INFO] Dashboard generated: {output_path}")
    print(f"[INFO] Source directory: {source_dir}")


if __name__ == "__main__":
    main()

