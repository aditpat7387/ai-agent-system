from pathlib import Path
import pandas as pd
import numpy as np
import duckdb
import yaml
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import plotly.graph_objects as go
from plotly.subplots import make_subplots

with open("configs/data_sources.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

db_path = cfg["storage"]["db_path"]
out = Path("artifacts/backtests")
out.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(db_path)

required_tables = [
    "ethusd_paper_trades_v7",
    "ethusd_predictions_calibrated_v7",
]

for table_name in required_tables:
    exists = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()[0]
    if not exists:
        raise ValueError(f"Missing table in {db_path}: {table_name}")

trades = con.execute("SELECT * FROM ethusd_paper_trades_v7").fetchdf()
cal = con.execute("SELECT * FROM ethusd_predictions_calibrated_v7").fetchdf()
con.close()

plot_df = pd.DataFrame({
    "raw_pred_proba": cal["pred_proba"].astype(float),
    "cal_pred_proba": cal["cal_pred_proba"].astype(float),
})

fixed_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
fixed_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

trades["raw_bucket"] = pd.cut(
    trades["raw_pred_proba"],
    bins=fixed_bins,
    labels=fixed_labels,
    include_lowest=True,
    right=True,
)

trades["cal_bucket"] = pd.cut(
    trades["cal_pred_proba"],
    bins=fixed_bins,
    labels=fixed_labels,
    include_lowest=True,
    right=True,
)

raw_bucket_perf = trades.groupby("raw_bucket", observed=False).agg(
    trades=("net_return", "size"),
    avg_pred_proba=("raw_pred_proba", "mean"),
    avg_net_return=("net_return", "mean"),
    median_net_return=("net_return", "median"),
    win_rate=("net_return", lambda s: (s > 0).mean()),
    total_net_return=("net_return", "sum"),
).reset_index()

cal_bucket_perf = trades.groupby("cal_bucket", observed=False).agg(
    trades=("net_return", "size"),
    avg_pred_proba=("cal_pred_proba", "mean"),
    avg_net_return=("net_return", "mean"),
    median_net_return=("net_return", "median"),
    win_rate=("net_return", lambda s: (s > 0).mean()),
    total_net_return=("net_return", "sum"),
).reset_index()

ks_stat, ks_pvalue = ks_2samp(plot_df["raw_pred_proba"], plot_df["cal_pred_proba"])

bins = np.linspace(0.0, 1.0, 51)
raw_hist, _ = np.histogram(plot_df["raw_pred_proba"], bins=bins, density=True)
cal_hist, _ = np.histogram(plot_df["cal_pred_proba"], bins=bins, density=True)

raw_hist = raw_hist + 1e-12
cal_hist = cal_hist + 1e-12
raw_hist = raw_hist / raw_hist.sum()
cal_hist = cal_hist / cal_hist.sum()

js_distance = float(jensenshannon(raw_hist, cal_hist, base=2))
js_divergence = float(js_distance ** 2)

distribution_metrics = pd.DataFrame([{
    "n_predictions": int(len(plot_df)),
    "raw_mean": float(plot_df["raw_pred_proba"].mean()),
    "cal_mean": float(plot_df["cal_pred_proba"].mean()),
    "raw_std": float(plot_df["raw_pred_proba"].std()),
    "cal_std": float(plot_df["cal_pred_proba"].std()),
    "raw_min": float(plot_df["raw_pred_proba"].min()),
    "cal_min": float(plot_df["cal_pred_proba"].min()),
    "raw_max": float(plot_df["raw_pred_proba"].max()),
    "cal_max": float(plot_df["cal_pred_proba"].max()),
    "ks_statistic": float(ks_stat),
    "ks_pvalue": float(ks_pvalue),
    "js_distance": js_distance,
    "js_divergence": js_divergence,
}])

plot_df.to_csv(out / "calibration_distribution_data_v7.csv", index=False)
raw_bucket_perf.to_csv(out / "raw_probability_bucket_performance_v7.csv", index=False)
cal_bucket_perf.to_csv(out / "calibrated_probability_bucket_performance_v7.csv", index=False)
distribution_metrics.to_csv(out / "distribution_comparison_metrics_v7.csv", index=False)

fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        "Raw vs calibrated probability histograms",
        "Raw vs calibrated probability box plots",
        "Raw bucket trade counts",
        "Calibrated bucket trade counts",
        "Raw bucket average net return",
        "Calibrated bucket average net return",
    ),
)

fig.add_trace(go.Histogram(x=plot_df["raw_pred_proba"], nbinsx=30, name="Raw", opacity=0.65), row=1, col=1)
fig.add_trace(go.Histogram(x=plot_df["cal_pred_proba"], nbinsx=30, name="Calibrated", opacity=0.65), row=1, col=1)

fig.add_trace(go.Box(y=plot_df["raw_pred_proba"], name="Raw", boxmean=True), row=1, col=2)
fig.add_trace(go.Box(y=plot_df["cal_pred_proba"], name="Calibrated", boxmean=True), row=1, col=2)

fig.add_trace(go.Bar(x=raw_bucket_perf["raw_bucket"].astype(str), y=raw_bucket_perf["trades"], name="Raw count"), row=2, col=1)
fig.add_trace(go.Bar(x=cal_bucket_perf["cal_bucket"].astype(str), y=cal_bucket_perf["trades"], name="Cal count"), row=2, col=2)

fig.add_trace(go.Bar(x=raw_bucket_perf["raw_bucket"].astype(str), y=raw_bucket_perf["avg_net_return"], name="Raw avg return"), row=3, col=1)
fig.add_trace(go.Bar(x=cal_bucket_perf["cal_bucket"].astype(str), y=cal_bucket_perf["avg_net_return"], name="Cal avg return"), row=3, col=2)

fig.update_layout(
    title=(
        "Probability distribution and fixed-bucket analysis v7"
        f"<br><sup>KS={ks_stat:.6f}, p={ks_pvalue:.6g}, "
        f"JS distance={js_distance:.6f}, JS divergence={js_divergence:.6f}</sup>"
    ),
    height=1200,
    width=1400,
    barmode="group",
)

fig.write_image(out / "probability_distribution_fixed_bucket_analysis_v7.png")

print("Distribution comparison metrics:")
print(distribution_metrics.to_string(index=False))

print("\nRaw bucket performance:")
print(raw_bucket_perf.to_string(index=False))

print("\nCalibrated bucket performance:")
print(cal_bucket_perf.to_string(index=False))

print(f"\nSaved: {out / 'calibration_distribution_data_v7.csv'}")
print(f"Saved: {out / 'raw_probability_bucket_performance_v7.csv'}")
print(f"Saved: {out / 'calibrated_probability_bucket_performance_v7.csv'}")
print(f"Saved: {out / 'distribution_comparison_metrics_v7.csv'}")
print(f"Saved: {out / 'probability_distribution_fixed_bucket_analysis_v7.png'}")