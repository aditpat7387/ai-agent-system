from pathlib import Path
import pandas as pd

REPORTS_DIR = Path("models/reports")


def load_mean_metrics(path):
    df = pd.read_csv(path)
    return df.mean(numeric_only=True)


def main():
    files = {
        "v1_logistic": REPORTS_DIR / "logistic_regression_walk_forward_metrics.csv",
        "v1_hgb": REPORTS_DIR / "hist_gradient_boosting_walk_forward_metrics.csv",
        "v2_logistic_binary": REPORTS_DIR / "logistic_regression_binary_walk_forward_metrics.csv",
        "v2_hgb_binary": REPORTS_DIR / "hist_gradient_boosting_binary_walk_forward_metrics.csv",
        "v2_logistic_multi": REPORTS_DIR / "logistic_regression_multiclass_walk_forward_metrics.csv",
        "v2_hgb_multi": REPORTS_DIR / "hist_gradient_boosting_multiclass_walk_forward_metrics.csv",
    }

    rows = []
    for name, path in files.items():
        if path.exists():
            metrics = load_mean_metrics(path)
            metrics["model_set"] = name
            rows.append(metrics)

    summary = pd.DataFrame(rows)
    print(summary)

    out = REPORTS_DIR / "baseline_comparison_v1_v2.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved comparison to {out}")


if __name__ == "__main__":
    main()