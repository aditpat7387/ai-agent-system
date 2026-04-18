from pathlib import Path
import pandas as pd


REPORTS_DIR = Path("models/reports")


def print_summary(model_name):
    metrics_path = REPORTS_DIR / f"{model_name}_walk_forward_metrics.csv"
    df = pd.read_csv(metrics_path)

    print(f"\n=== {model_name} ===")
    print(df.describe(include="all"))

    print("\nMean metrics:")
    print(df[["accuracy", "precision", "recall", "f1", "roc_auc"]].mean())


def main():
    for model_name in ["logistic_regression", "hist_gradient_boosting"]:
        print_summary(model_name)


if __name__ == "__main__":
    main()