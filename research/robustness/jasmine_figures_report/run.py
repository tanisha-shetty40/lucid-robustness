
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from .bootstrap import bootstrap_metric
from .figures import plot_metric_vs_severity, create_synthetic_gallery


def accuracy_metric(df):
    predictions = (
        df["prob_pneumonia"].to_numpy() >= 0.5
    ).astype(int)

    labels = df["label"].to_numpy()

    return float(np.mean(predictions == labels))

def auc_metric(df):
    from scipy.stats import rankdata

    labels = df["label"].to_numpy()
    scores = df["prob_pneumonia"].to_numpy()

    positives = labels == 1
    negatives = labels == 0

    n_pos = np.sum(positives)
    n_neg = np.sum(negatives)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = rankdata(scores)

    auc = (
        np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2
    ) / (n_pos * n_neg)

    return float(auc)


def main():
    parser = argparse.ArgumentParser(
        description="Lucid robustness analysis"
    )

    parser.add_argument("--sweep", type=str)
    parser.add_argument("--demo", action="store_true")

    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent

    if args.demo:
        from .make_demo_data import generate_demo_data
        df = generate_demo_data()
        print("Using synthetic demo data.")

    elif args.sweep:
        df = pd.read_csv(args.sweep)
        print("Using supplied sweep data.")

    else:
        parser.error("Provide either --sweep or --demo.")

    required = {
        "image_id",
        "label",
        "corruption",
        "severity",
        "prob_pneumonia",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    group_cols = ["corruption", "severity"]

    accuracy_ci = bootstrap_metric(
        df, accuracy_metric, group_cols,
        n_boot=500, seed=42
    )

    auc_ci = bootstrap_metric(
        df, auc_metric, group_cols,
        n_boot=500, seed=42
    )

    plot_metric_vs_severity(
        accuracy_ci,
        "Accuracy",
        output_dir / "fig_accuracy_vs_severity.png"
    )

    plot_metric_vs_severity(
        auc_ci,
        "ROC-AUC",
        output_dir / "fig_auc_vs_severity.png"
    )

    accuracy_ci.to_csv(
        output_dir / "accuracy_summary.csv",
        index=False
    )

    auc_ci.to_csv(
        output_dir / "auc_summary.csv",
        index=False
    )

   
    create_synthetic_gallery(
        output_dir / "fig_gallery_synthetic.png"
    )

    print("Robustness analysis completed.")
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()
