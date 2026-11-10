
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


ROOT = Path.cwd()

INPUT = (
    ROOT / "results" / "robustness" / "real_benchmark"
    / "quality_prediction_analysis.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "robustness" / "real_benchmark"
)

def main():
    print("Loading image-quality analysis...")
    df = pd.read_csv(INPUT)

    required = {
        "image_id", "corruption", "severity",
        "mse", "psnr", "ssim", "prediction_error"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Exclude baseline because its quality metrics are constant.
    corrupted = df[df["corruption"] != "none"].copy()

    print("Calculating correlations by corruption and severity...")

    correlation_rows = []

    for (corruption, severity), group in corrupted.groupby(
        ["corruption", "severity"]
    ):
        for metric in ["psnr", "ssim", "mse"]:
            result = spearmanr(
                group[metric],
                group["prediction_error"]
            )

            correlation_rows.append({
                "corruption": corruption,
                "severity": severity,
                "metric": metric,
                "spearman_rho": result.statistic,
                "p_value": result.pvalue,
                "sample_count": len(group)
            })

    correlations = pd.DataFrame(correlation_rows)

    correlations.to_csv(
        OUTPUT_DIR / "quality_error_correlations.csv",
        index=False
    )

    print("Creating condition-level summary...")

    summary = df.groupby(
        ["corruption", "severity"]
    ).agg(
        mean_psnr=("psnr", "mean"),
        mean_ssim=("ssim", "mean"),
        mean_mse=("mse", "mean"),
        prediction_errors=("prediction_error", "sum"),
        total_images=("image_id", "count")
    ).reset_index()

    summary["error_rate_percent"] = (
        summary["prediction_errors"]
        / summary["total_images"] * 100
    )

    summary.to_csv(
        OUTPUT_DIR / "quality_error_condition_summary.csv",
        index=False
    )

    # Plot 1: Prediction error rate across severity levels.
    plot_data = summary[summary["corruption"] != "none"]

    plt.figure(figsize=(10, 6))

    for corruption, group in plot_data.groupby("corruption"):
        group = group.sort_values("severity")
        plt.plot(
            group["severity"],
            group["error_rate_percent"],
            marker="o",
            label=corruption
        )

    plt.xlabel("Corruption severity")
    plt.ylabel("Prediction error rate (%)")
    plt.title("Prediction Error Rate vs Corruption Severity")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "error_rate_vs_severity.png",
        dpi=300
    )
    plt.close()

    # Plot 2: Mean SSIM vs prediction error rate.
    plt.figure(figsize=(9, 6))

    plt.scatter(
        plot_data["mean_ssim"],
        plot_data["error_rate_percent"]
    )

    for row in plot_data.itertuples():
        plt.annotate(
            f"{row.corruption}-{row.severity}",
            (row.mean_ssim, row.error_rate_percent),
            fontsize=6,
            alpha=0.7
        )

    plt.xlabel("Mean SSIM")
    plt.ylabel("Prediction error rate (%)")
    plt.title("Image Structural Quality vs Prediction Errors")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "ssim_vs_error_rate.png",
        dpi=300
    )
    plt.close()

    print("\nCorrelation analysis completed.")
    print("Correlation records:", len(correlations))
    print("Condition summaries:", len(summary))
    print("Files saved in:", OUTPUT_DIR)

    print("\nSample correlations:")
    print(correlations.head(12).to_string(index=False))

    print(
        "\nNote: Correlations are calculated separately "
        "for each corruption and severity across 874 images. "
        "P-values are exploratory and do not establish causation."
    )


if __name__ == "__main__":
    main()
