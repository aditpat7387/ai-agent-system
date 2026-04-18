from pathlib import Path
import os
import pandas as pd
import duckdb
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FIXED_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FIXED_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_db_path(root: Path) -> str:
    cfg_path = root / "configs" / "data_sources.yaml"
    print(f"[INFO] Loading config from: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_path = cfg["storage"]["db_path"]
    print(f"[INFO] Config DB path: {db_path}")
    return db_path


def ensure_tables(con, tables):
    missing = []
    for table_name in tables:
        exists = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()[0]
        print(f"[INFO] Table check - {table_name}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(table_name)

    if missing:
        raise ValueError(f"Missing tables: {missing}")


def bucket_stats(df, bucket_col, group_name, prob_col):
    out = df.groupby(bucket_col, observed=False).agg(
        trades=("net_return", "size"),
        avg_pred_proba=(prob_col, "mean"),
        avg_net_return=("net_return", "mean"),
        median_net_return=("net_return", "median"),
        win_rate=("net_return", lambda s: (s > 0).mean()),
        total_net_return=("net_return", "sum"),
    ).reset_index()

    out.insert(0, "group_name", group_name)
    out.rename(columns={bucket_col: "bucket"}, inplace=True)
    return out


def export_figure(fig, out_dir: Path, stem: str):
    png_path = out_dir / f"{stem}.png"
    html_path = out_dir / f"{stem}.html"

    try:
        print(f"[INFO] Attempting PNG export: {png_path}")
        fig.write_image(png_path)
        print(f"[INFO] Saved PNG: {png_path}")
        return png_path
    except Exception as e:
        print(f"[WARN] PNG export failed: {e}")
        print(f"[INFO] Falling back to HTML export: {html_path}")
        fig.write_html(html_path)
        print(f"[INFO] Saved HTML: {html_path}")
        return html_path


def main():
    root = project_root()
    print(f"[INFO] Current working directory: {os.getcwd()}")
    print(f"[INFO] Project root resolved to: {root}")

    out_dir = root / "artifacts" / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    db_path = load_db_path(root)
    db_full_path = Path(db_path)
    if not db_full_path.is_absolute():
        db_full_path = root / db_path
    print(f"[INFO] Resolved DB path: {db_full_path}")

    con = duckdb.connect(str(db_full_path))

    ensure_tables(
        con,
        [
            "ethusd_paper_trades_v7",
            "ethusd_predictions_calibrated_v7",
        ],
    )

    print("[INFO] Reading ethusd_paper_trades_v7...")
    trades = con.execute("SELECT * FROM ethusd_paper_trades_v7").fetchdf()
    print(f"[INFO] Trades rows: {len(trades)}")

    print("[INFO] Reading ethusd_predictions_calibrated_v7...")
    cal = con.execute("SELECT * FROM ethusd_predictions_calibrated_v7").fetchdf()
    print(f"[INFO] Prediction rows: {len(cal)}")

    con.close()

    trades["cal_pred_proba"] = trades["cal_pred_proba"].astype(float)
    trades["raw_pred_proba"] = trades["raw_pred_proba"].astype(float)
    trades["regime_label"] = trades["regime_label"].astype(str)
    trades["side"] = trades["side"].astype(str)

    trades["cal_bucket"] = pd.cut(
        trades["cal_pred_proba"],
        bins=FIXED_BINS,
        labels=FIXED_LABELS,
        include_lowest=True,
        right=True,
    )

    trades["raw_bucket"] = pd.cut(
        trades["raw_pred_proba"],
        bins=FIXED_BINS,
        labels=FIXED_LABELS,
        include_lowest=True,
        right=True,
    )

    print("[INFO] Building aggregate bucket summaries...")
    all_cal = bucket_stats(trades, "cal_bucket", "all_regimes", "cal_pred_proba")
    all_raw = bucket_stats(trades, "raw_bucket", "all_regimes_raw", "raw_pred_proba")

    regime_frames = []
    for regime in sorted(trades["regime_label"].dropna().unique()):
        sub = trades[trades["regime_label"] == regime].copy()
        print(f"[INFO] Regime {regime}: {len(sub)} trades")
        if not sub.empty:
            sub["regime_bucket"] = pd.cut(
                sub["cal_pred_proba"],
                bins=FIXED_BINS,
                labels=FIXED_LABELS,
                include_lowest=True,
                right=True,
            )
            regime_frames.append(
                bucket_stats(sub, "regime_bucket", regime, "cal_pred_proba")
            )

    side_frames = []
    for side in sorted(trades["side"].dropna().unique()):
        sub = trades[trades["side"] == side].copy()
        print(f"[INFO] Side {side}: {len(sub)} trades")
        if not sub.empty:
            sub["side_bucket"] = pd.cut(
                sub["cal_pred_proba"],
                bins=FIXED_BINS,
                labels=FIXED_LABELS,
                include_lowest=True,
                right=True,
            )
            side_frames.append(
                bucket_stats(sub, "side_bucket", side, "cal_pred_proba")
            )

    bucket_summary = pd.concat(
        [all_cal, all_raw] + regime_frames + side_frames,
        ignore_index=True,
    )

    bucket_summary_path = out_dir / "regime_side_probability_bucket_summary_v7.csv"
    bucket_summary.to_csv(bucket_summary_path, index=False)
    print(f"[INFO] Saved: {bucket_summary_path}")

    trade_counts = trades.groupby(["regime_label", "side"]).agg(
        trades=("net_return", "size")
    ).reset_index()

    trade_counts_path = out_dir / "regime_side_trade_counts_v7.csv"
    trade_counts.to_csv(trade_counts_path, index=False)
    print(f"[INFO] Saved: {trade_counts_path}")

    win_rates = trades.groupby(["regime_label", "side"]).agg(
        win_rate=("net_return", lambda s: (s > 0).mean())
    ).reset_index()

    print("[INFO] Building figure...")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "All trades: calibrated probability buckets",
            "All trades: raw probability buckets",
            "Trade counts by regime and side",
            "Win rate by regime and side",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=all_cal["bucket"].astype(str),
            y=all_cal["trades"],
            name="Cal bucket trades",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=all_raw["bucket"].astype(str),
            y=all_raw["trades"],
            name="Raw bucket trades",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=trade_counts["regime_label"] + " / " + trade_counts["side"],
            y=trade_counts["trades"],
            name="Trade counts",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=win_rates["regime_label"] + " / " + win_rates["side"],
            y=win_rates["win_rate"],
            name="Win rate",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=900,
        width=1300,
        barmode="group",
        title="Regime and probability band analysis v7",
    )

    chart_path = export_figure(fig, out_dir, "regime_side_probability_band_analysis_v7")

    print("\n[INFO] Bucket summary:")
    print(bucket_summary.to_string(index=False))

    print("\n[INFO] Trade counts:")
    print(trade_counts.to_string(index=False))

    print(f"\n[INFO] Final chart output: {chart_path}")


if __name__ == "__main__":
    main()