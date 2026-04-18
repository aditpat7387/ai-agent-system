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
    ensure_tables(con, ["ethusd_predictions_calibrated_v7"])

    print("[INFO] Reading ethusd_predictions_calibrated_v7...")
    df = con.execute("SELECT * FROM ethusd_predictions_calibrated_v7").fetchdf()
    con.close()
    print(f"[INFO] Prediction rows: {len(df)}")

    required_cols = ["regime_label", "pred_label", "event_label", "pred_proba", "cal_pred_proba"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df["regime_label"] = df["regime_label"].astype(str)
    df["pred_label"] = df["pred_label"].astype(int)
    df["event_label"] = df["event_label"].astype(int)
    df["pred_proba"] = df["pred_proba"].astype(float)
    df["cal_pred_proba"] = df["cal_pred_proba"].astype(float)

    df["raw_bucket"] = pd.cut(
        df["pred_proba"],
        bins=FIXED_BINS,
        labels=FIXED_LABELS,
        include_lowest=True,
        right=True,
    )

    df["cal_bucket"] = pd.cut(
        df["cal_pred_proba"],
        bins=FIXED_BINS,
        labels=FIXED_LABELS,
        include_lowest=True,
        right=True,
    )

    print("[INFO] Building summaries...")

    regime_pred_summary = (
        df.groupby(["regime_label", "pred_label"])
        .agg(
            rows=("pred_label", "size"),
            avg_raw_pred_proba=("pred_proba", "mean"),
            avg_cal_pred_proba=("cal_pred_proba", "mean"),
            positive_rate=("event_label", "mean"),
        )
        .reset_index()
    )

    regime_actual_summary = (
        df.groupby(["regime_label", "event_label"])
        .agg(rows=("event_label", "size"))
        .reset_index()
    )

    confusion_summary = (
        df.groupby(["regime_label", "event_label", "pred_label"])
        .agg(rows=("pred_label", "size"))
        .reset_index()
        .sort_values(["regime_label", "event_label", "pred_label"])
    )

    raw_bucket_summary = (
        df.groupby(["regime_label", "pred_label", "raw_bucket"], observed=False)
        .agg(
            rows=("pred_label", "size"),
            actual_positive_rate=("event_label", "mean"),
        )
        .reset_index()
    )

    cal_bucket_summary = (
        df.groupby(["regime_label", "pred_label", "cal_bucket"], observed=False)
        .agg(
            rows=("pred_label", "size"),
            actual_positive_rate=("event_label", "mean"),
        )
        .reset_index()
    )

    overall_pred_counts = (
        df.groupby("pred_label")
        .agg(rows=("pred_label", "size"))
        .reset_index()
    )

    overall_actual_counts = (
        df.groupby("event_label")
        .agg(rows=("event_label", "size"))
        .reset_index()
    )

    regime_pred_summary.to_csv(out_dir / "regime_pred_summary_v7.csv", index=False)
    regime_actual_summary.to_csv(out_dir / "regime_actual_summary_v7.csv", index=False)
    confusion_summary.to_csv(out_dir / "regime_confusion_summary_v7.csv", index=False)
    raw_bucket_summary.to_csv(out_dir / "regime_raw_probability_bucket_summary_v7.csv", index=False)
    cal_bucket_summary.to_csv(out_dir / "regime_cal_probability_bucket_summary_v7.csv", index=False)
    overall_pred_counts.to_csv(out_dir / "overall_pred_counts_v7.csv", index=False)
    overall_actual_counts.to_csv(out_dir / "overall_actual_counts_v7.csv", index=False)

    print(f"[INFO] Saved: {out_dir / 'regime_pred_summary_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'regime_actual_summary_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'regime_confusion_summary_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'regime_raw_probability_bucket_summary_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'regime_cal_probability_bucket_summary_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'overall_pred_counts_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'overall_actual_counts_v7.csv'}")

    print("[INFO] Building figure...")
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Overall predicted class counts",
            "Overall actual class counts",
            "Predicted class counts by regime",
            "Average calibrated probability by regime and predicted class",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=overall_pred_counts["pred_label"].astype(str),
            y=overall_pred_counts["rows"],
            name="Predicted class count",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=overall_actual_counts["event_label"].astype(str),
            y=overall_actual_counts["rows"],
            name="Actual class count",
        ),
        row=1,
        col=2,
    )

    for pred_label in sorted(regime_pred_summary["pred_label"].unique()):
        sub = regime_pred_summary[regime_pred_summary["pred_label"] == pred_label]
        fig.add_trace(
            go.Bar(
                x=sub["regime_label"],
                y=sub["rows"],
                name=f"Pred {pred_label} count",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=sub["regime_label"],
                y=sub["avg_cal_pred_proba"],
                name=f"Pred {pred_label} avg cal prob",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=900,
        width=1400,
        barmode="group",
        title="All prediction diagnostics v7",
    )

    chart_path = export_figure(fig, out_dir, "all_prediction_diagnostics_v7")

    print("\n[INFO] Regime / predicted-label summary:")
    print(regime_pred_summary.to_string(index=False))

    print("\n[INFO] Regime / actual-label summary:")
    print(regime_actual_summary.to_string(index=False))

    print("\n[INFO] Regime confusion summary:")
    print(confusion_summary.to_string(index=False))

    print(f"\n[INFO] Final chart output: {chart_path}")


if __name__ == "__main__":
    main()