
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "mini_sweep_results.csv"
OUTPUT_CSV = BASE_DIR / "summary.csv"

REQUIRED_COLUMNS = [
    "image_id",
    "label",
    "corruption",
    "severity",
    "prob_pneumonia",
]


def calculate_metrics(group):
    labels = group["label"].astype(int)
    probabilities = group["prob_pneumonia"].astype(float)

    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(labels, predictions)

    precision = precision_score(
        labels, predictions, zero_division=0
    )

    sensitivity = recall_score(
        labels, predictions, zero_division=0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else np.nan
    )

    f1 = f1_score(
        labels, predictions, zero_division=0
    )

    # AUC requires both positive and negative labels.
    if labels.nunique() == 2:
        auc = roc_auc_score(labels, probabilities)
    else:
        auc = np.nan

    return {
        "n": len(group),
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "error_count": fp + fn,
        "error_rate": (fp + fn) / len(group),
        "average_probability": probabilities.mean(),
        "average_quality": (
            group["quality"].mean()
            if "quality" in group.columns
            else np.nan
        ),
    }


def summarize_results(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must contain only 0 or 1.")

    if not df["prob_pneumonia"].between(0, 1).all():
        raise ValueError(
            "Prediction probabilities must be between 0 and 1."
        )

    summary_rows = []

    for (corruption, severity), group in df.groupby(
        ["corruption", "severity"]
    ):
        metrics = calculate_metrics(group)

        summary_rows.append({
            "corruption": corruption,
            "severity": int(severity),
            **metrics,
        })

    summary = pd.DataFrame(summary_rows)

    # Compare each condition with baseline accuracy.
    baseline = summary[
        (summary["corruption"] == "baseline")
        & (summary["severity"] == 0)
    ]

    if not baseline.empty:
        baseline_accuracy = baseline.iloc[0]["accuracy"]

        summary["accuracy_drop"] = (
            baseline_accuracy - summary["accuracy"]
        )
    else:
        summary["accuracy_drop"] = np.nan

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(output_csv, index=False)

    print("Summary generated successfully!")
    print("Conditions:", len(summary))
    print("Saved to:", output_csv)

    return summary


if __name__ == "__main__":
    summarize_results()
