
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def summarize_results(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required_columns = {
        "image_id",
        "label",
        "corruption",
        "severity",
        "probability",
        "prediction",
        "correct",
        "quality_score",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    for (corruption, severity), group in df.groupby(
        ["corruption", "severity"]
    ):
        y_true = group["label"].astype(int)
        y_pred = group["prediction"].astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        accuracy = (tp + tn) / len(group)
        precision = precision_score(
            y_true, y_pred, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        specificity = tn / (tn + fp) if (tn + fp) else float("nan")

        results.append({
            "corruption": corruption,
            "severity": severity,
            "total_images": group["image_id"].nunique(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "error_rate": 1 - accuracy,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "average_probability": group["probability"].mean(),
            "average_quality": group["quality_score"].mean(),
        })

    summary = pd.DataFrame(results)

    baseline = summary[
        summary["corruption"] == "baseline"
    ]

    if baseline.empty:
        raise ValueError("Baseline results are missing.")

    baseline_accuracy = baseline["accuracy"].iloc[0]

    summary["accuracy_drop"] = (
        baseline_accuracy - summary["accuracy"]
    )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_path, index=False)

    print("Summary generated successfully!")
    print(summary.to_string(index=False))
    print("Saved to:", output_path)


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent

    input_file = folder / "brahmi_corruption_demo_results.csv"
    output_file = folder / "demo_summary.csv"

    summarize_results(input_file, output_file)
