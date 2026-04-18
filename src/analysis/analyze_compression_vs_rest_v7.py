from pathlib import Path
import os
import pandas as pd
import duckdb
import yaml
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    df = df.copy()
    df["regime_group"] = np.where(df["regime_label"].astype(str) == "compression", "compression", "rest")
    df["pred_label"] = df["pred_label"].astype(int)
    df["event_label"] = df["event_label"].astype(int)
    df["pred_proba"] = df["pred_proba"].astype(float)
    df["cal_pred_proba"] = df["cal_pred_proba"].astype(float)

    summary = df.groupby("regime_group").agg(
        rows=("regime_group", "size"),
        pred0=("pred_label", lambda s: (s == 0).sum()),
        pred1=("pred_label", lambda s: (s == 1).sum()),
        actual0=("event_label", lambda s: (s == 0).sum()),
        actual1=("event_label", lambda s: (s == 1).sum()),
        avg_raw_pred_proba=("pred_proba", "mean"),
        avg_cal_pred_proba=("cal_pred_proba", "mean"),
        pos_rate=("event_label", "mean"),
    ).reset_index()

    by_group_label = df.groupby(["regime_group", "pred_label"]).agg(
        rows=("pred_label", "size"),
        avg_raw_pred_proba=("pred_proba", "mean"),
        avg_cal_pred_proba=("cal_pred_proba", "mean"),
        positive_rate=("event_label", "mean"),
    ).reset_index()

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {"pred_label", "event_label", "pred_proba", "cal_pred_proba", "fold"}
    ]
    feature_rows = []
    for col in numeric_cols:
        for grp in ["compression", "rest"]:
            sub = df[df["regime_group"] == grp]
            g0 = sub[sub["pred_label"] == 0][col].astype(float)
            g1 = sub[sub["pred_label"] == 1][col].astype(float)
            if len(g0) and len(g1):
                feature_rows.append({
                    "regime_group": grp,
                    "feature": col,
                    "pred0_count": int(len(g0)),
                    "pred1_count": int(len(g1)),
                    "pred0_mean": float(g0.mean()),
                    "pred1_mean": float(g1.mean()),
                    "mean_diff_pred1_minus_pred0": float(g1.mean() - g0.mean()),
                })
    feature_balance = pd.DataFrame(feature_rows)

    summary.to_csv(out_dir / "compression_vs_rest_summary_v7.csv", index=False)
    by_group_label.to_csv(out_dir / "compression_vs_rest_label_summary_v7.csv", index=False)
    feature_balance.to_csv(out_dir / "compression_vs_rest_feature_balance_v7.csv", index=False)

    print(f"[INFO] Saved: {out_dir / 'compression_vs_rest_summary_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'compression_vs_rest_label_summary_v7.csv'}")
    print(f"[INFO] Saved: {out_dir / 'compression_vs_rest_feature_balance_v7.csv'}")

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Prediction counts by group",
        "Actual counts by group",
        "Avg calibrated probability by group and predicted class",
        "Avg raw probability by group and predicted class",
    ))

    for pred_label in sorted(by_group_label["pred_label"].unique()):
        sub = by_group_label[by_group_label["pred_label"] == pred_label]
        fig.add_trace(go.Bar(x=sub["regime_group"], y=sub["rows"], name=f"Pred {pred_label} count"), row=1, col=1)
        fig.add_trace(go.Bar(x=sub["regime_group"], y=sub["avg_cal_pred_proba"], name=f"Pred {pred_label} avg cal"), row=2, col=1)
        fig.add_trace(go.Bar(x=sub["regime_group"], y=sub["avg_raw_pred_proba"], name=f"Pred {pred_label} avg raw"), row=2, col=2)

    actual_counts = df.groupby(["regime_group", "event_label"]).size().reset_index(name="rows")
    for event_label in sorted(actual_counts["event_label"].unique()):
        sub = actual_counts[actual_counts["event_label"] == event_label]
        fig.add_trace(go.Bar(x=sub["regime_group"], y=sub["rows"], name=f"Actual {event_label} count"), row=1, col=2)

    fig.update_layout(height=900, width=1500, barmode="group", title="Compression vs rest diagnostics v7")
    chart_path = export_figure(fig, out_dir, "compression_vs_rest_diagnostics_v7")

    print("\n[INFO] Summary:")
    print(summary.to_string(index=False))
    print("\n[INFO] Label summary:")
    print(by_group_label.to_string(index=False))
    if not feature_balance.empty:
        print("\n[INFO] Feature balance preview:")
        print(feature_balance.head(50).to_string(index=False))
    else:
        print("\n[INFO] Feature balance preview: no overlapping classes within groups")

    print(f"\n[INFO] Final chart output: {chart_path}")

if __name__ == "__main__":
    main()