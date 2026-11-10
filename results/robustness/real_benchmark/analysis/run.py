import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .bootstrap import bootstrap_metric
from .figures import plot_metric_vs_severity


def predictions(df):
    return (df["prob_pneumonia"].to_numpy() >= 0.5).astype(int)


def accuracy_metric(df):
    return float(np.mean(predictions(df) == df["label"].to_numpy()))


def sensitivity_metric(df):
    labels = df["label"].to_numpy()
    pred = predictions(df)
    positives = labels == 1
    return float(np.sum(pred[positives] == 1) / np.sum(positives)) if np.sum(positives) else np.nan


def specificity_metric(df):
    labels = df["label"].to_numpy()
    pred = predictions(df)
    negatives = labels == 0
    return float(np.sum(pred[negatives] == 0) / np.sum(negatives)) if np.sum(negatives) else np.nan


def auc_metric(df):
    labels = df["label"].to_numpy()
    scores = df["prob_pneumonia"].to_numpy()
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def main():
    parser = argparse.ArgumentParser(description="LUCID real robustness analysis")
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent
    df = pd.read_csv(args.sweep)

    required = {
        "image_id", "label", "corruption", "severity", "prob_pneumonia"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Input CSV contains missing required values.")

    metrics = {
        "accuracy": accuracy_metric,
        "sensitivity": sensitivity_metric,
        "specificity": specificity_metric,
        "auc": auc_metric,
    }

    group_cols = ["corruption", "severity"]

    for name, metric_fn in metrics.items():
        print(f"Calculating {name} with 500 bootstrap resamples...")

        summary = bootstrap_metric(
            df, metric_fn, group_cols,
            n_boot=500, seed=42
        )

        summary.to_csv(output_dir / f"{name}_summary.csv", index=False)

        plot_metric_vs_severity(
            summary,
            name.upper() if name == "auc" else name.capitalize(),
            output_dir / f"fig_{name}_vs_severity.png"
        )

        print(summary.to_string(index=False))

    print("\nReal benchmark analysis completed.")
    print("Output folder:", output_dir)


if __name__ == "__main__":
    main()