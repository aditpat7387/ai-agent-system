from pathlib import Path
import os
import pandas as pd
import duckdb
import yaml
import numpy as np
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


def safe_std(s):
    v = s.astype(float).std()
    return float(v) if pd.notna(v) else np.nan


def summarize_regime(df, regime_name):
    sub = df[df["regime_label"] == regime_name].copy()
    if sub.empty:
        return None

    numeric_cols = [
        c for c in sub.columns
        if c not in {
            "open_time", "close_time", "regime_label", "pred_label", "event_label",
            "pred_proba", "cal_pred_proba", "raw_bucket", "cal_bucket"
        } and pd.api.types.is_numeric_dtype(sub[c])
    ]

    rows = []
    for col in numeric_cols:
        grp0 = sub[sub["pred_label"] == 0][col].astype(float)
        grp1 = sub[sub["pred_label"] == 1][col].astype(float)

        if len(grp0) == 0 or len(grp1) == 0:
            continue

        rows.append({
            "regime_label": regime_name,
            "feature": col,
            "pred0_count": int(len(grp0)),
            "pred1_count": int(len(grp1)),
            "pred0_mean": float(grp0.mean()),
            "pred1_mean": float(grp1.mean()),
            "mean_diff_pred1_minus_pred0": float(grp1.mean() - grp0.mean()),
            "pred0_std": safe_std(grp0),
            "pred1_std": safe_std(grp1),
        })

    return pd.DataFrame(rows) if rows else None


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
    df["raw_bucket"] = pd.cut(df["pred_proba"], bins=FIXED_BINS, labels=FIXED_LABELS, include_lowest=True)
    df["cal_bucket"] = pd.cut(df["cal_pred_proba"], bins=FIXED_BINS, labels=FIXED_LABELS, include_lowest=True)

    print("[INFO] Building regime summaries...")

    regime_counts = df.groupby(["regime_label", "pred_label"]).agg(
        rows=("pred_label", "size"),
        avg_raw_pred_proba=("pred_proba", "mean"),
        avg_cal_pred_proba=("cal_pred_proba", "mean"),
        positive_rate=("event_label", "mean"),
    ).reset_index()

    regime_event_counts = df.groupby(["regime_label", "event_label"]).agg(
        rows=("event_label", "size")
    ).reset_index()

    separation_rows = []
    for regime in sorted(df["regime_label"].dropna().unique()):
        part = summarize_regime(df, regime)
        if part is not None:
            separation_rows.append(part)

    if separation_rows:
        feature_balance = pd.concat(separation_rows, ignore_index=True)
    else:
        feature_balance = pd.DataFrame()

    regime_counts.to_csv(out_dir / "regime_class_counts_v7.csv", index=False)
    regime_event_counts.to_csv(out_dir / "regime_event_counts_v7.csv", index=False)
    feature_balance.to_csv(out_dir / "regime_feature_balance_v7.csv", index=False)

    print(f"[INFO] Saved: {out_dir / 'regime_class_counts_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'regime_event_counts_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'regime_feature_balance_v7.csv'}")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Predicted class counts by regime",
            "Actual label counts by regime",
            "Average calibrated probability by regime and predicted class",
            "Average raw probability by regime and predicted class",
        ),
    )

    for pred_label in sorted(regime_counts["pred_label"].unique()):
        sub = regime_counts[regime_counts["pred_label"] == pred_label]
        fig.add_trace(
            go.Bar(
                x=sub["regime_label"],
                y=sub["rows"],
                name=f"Pred {pred_label} count",
            ),
            row=1,
            col=1,
        )

    for event_label in sorted(regime_event_counts["event_label"].unique()):
        sub = regime_event_counts[regime_event_counts["event_label"] == event_label]
        fig.add_trace(
            go.Bar(
                x=sub["regime_label"],
                y=sub["rows"],
                name=f"Actual {event_label} count",
            ),
            row=1,
            col=2,
        )

    for pred_label in sorted(regime_counts["pred_label"].unique()):
        sub = regime_counts[regime_counts["pred_label"] == pred_label]
        fig.add_trace(
            go.Bar(
                x=sub["regime_label"],
                y=sub["avg_cal_pred_proba"],
                name=f"Pred {pred_label} avg cal",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=sub["regime_label"],
                y=sub["avg_raw_pred_proba"],
                name=f"Pred {pred_label} avg raw",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=900,
        width=1500,
        barmode="group",
        title="Regime class balance and separation diagnostics v7",
    )

    chart_path = export_figure(fig, out_dir, "regime_class_balance_diagnostics_v7")

    print("\n[INFO] Regime class counts:")
    print(regime_counts.to_string(index=False))

    print("\n[INFO] Regime actual counts:")
    print(regime_event_counts.to_string(index=False))

    if not feature_balance.empty:
        print("\n[INFO] Feature balance preview:")
        print(feature_balance.head(50).to_string(index=False))
    else:
        print("\n[INFO] Feature balance preview: no numeric feature separation rows found")

    print(f"\n[INFO] Final chart output: {chart_path}")


if __name__ == "__main__":
    main()