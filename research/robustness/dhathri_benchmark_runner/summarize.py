from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent

INPUT_CSV = BASE_DIR / "sweep.csv"
OUTPUT_CSV = BASE_DIR / "summary.csv"

REQUIRED_COLUMNS = [
    "image_id", "label", "corruption", "severity",
    "prob_pneumonia", "quality",
    "q_brightness", "q_contrast", "q_sharpness",
    "q_noise", "q_clipping"
]


def calculate_metrics(group):
    labels = group["label"].astype(int)
    probabilities = group["prob_pneumonia"].astype(float)

    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    accuracy = (tn + tp) / len(group)

    sensitivity = (
        tp / (tp + fn) if (tp + fn) else np.nan
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) else np.nan
    )

    auc = (
        roc_auc_score(labels, probabilities)
        if labels.nunique() == 2
        else np.nan
    )

    return {
        "n": len(group),
        "auc": auc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def summarize_results(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    df["label"] = pd.to_numeric(df["label"], errors="raise")
    df["severity"] = pd.to_numeric(
        df["severity"], errors="raise"
    )
    df["prob_pneumonia"] = pd.to_numeric(
        df["prob_pneumonia"], errors="raise"
    )

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1.")

    if not df["prob_pneumonia"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    rows = []

    for (corruption, severity), group in df.groupby(
        ["corruption", "severity"]
    ):
        rows.append({
            "corruption": corruption,
            "severity": int(severity),
            **calculate_metrics(group),
        })

    summary = pd.DataFrame(rows)

    summary = summary[
        [
            "corruption", "severity", "n", "auc",
            "accuracy", "sensitivity", "specificity"
        ]
    ]

    summary = summary.sort_values(
        ["corruption", "severity"]
    )

    summary.to_csv(output_csv, index=False)

    print("Summary generated successfully!")
    print("Conditions:", len(summary))
    print("Columns:", list(summary.columns))
    print("Saved to:", output_csv)

    return summary


if __name__ == "__main__":
    summarize_results()
