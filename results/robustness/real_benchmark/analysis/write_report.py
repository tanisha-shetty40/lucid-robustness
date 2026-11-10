from pathlib import Path
import pandas as pd

analysis = Path(__file__).resolve().parent
root = analysis.parent

metrics = ["accuracy", "sensitivity", "specificity", "auc"]
corruptions = [
    "blur", "brightness_down", "brightness_up",
    "contrast", "downscale", "jpeg", "noise"
]

data = {}
for metric in metrics:
    df = pd.read_csv(analysis / f"{metric}_summary.csv")
    if len(df) != 36 or df[["value", "low", "high"]].isna().any().any():
        raise ValueError(f"{metric}: summary validation failed")
    data[metric] = df

def cell(row):
    return (
        f"{row['value']:.2%} "
        f"({row['low']:.2%}–{row['high']:.2%})"
    )

baseline = {}
for metric, df in data.items():
    row = df[(df.corruption == "none") & (df.severity == 0)].iloc[0]
    baseline[metric] = cell(row)

merged = None
for metric, df in data.items():
    part = df[df.corruption != "none"][
        ["corruption", "severity", "value", "low", "high"]
    ].copy()
    part = part.rename(columns={
        "value": f"{metric}_value",
        "low": f"{metric}_low",
        "high": f"{metric}_high"
    })
    merged = part if merged is None else merged.merge(
        part, on=["corruption", "severity"]
    )

lines = [
    "# LUCID — Real-Data Robustness Benchmark Report",
    "",
    "## 1. Objective",
    "",
    "Evaluate how controlled image-quality corruptions affect the performance "
    "of the frozen LUCID pneumonia-classification model.",
    "",
    "## 2. Experimental setup",
    "",
    "- Model: ResNet18, loaded from `models/classifier.pt`.",
    "- Input size: 224 × 224 pixels.",
    "- Classes: NORMAL and PNEUMONIA.",
    "- Positive class: PNEUMONIA.",
    "- Evaluation: original held-out test split, 874 images.",
    "- Preprocessing: grayscale converted to three channels, resized to "
    "224 × 224, converted to tensor, and normalized using ImageNet mean "
    "and standard deviation.",
    "- Model inference: evaluation mode on CPU.",
    "- Decision threshold: pneumonia probability >= 0.5.",
    "- Reproducibility: stable SHA-256-derived corruption seed using "
    "image ID, corruption, severity, and base seed 42.",
    "",
    "## 3. Corruption protocol",
    "",
    "Each of the seven corruption types was applied at five severity levels. "
    "Severity 0 represents the unchanged baseline.",
    "",
    "| Corruption | Severity parameters (1 to 5) |",
    "|---|---|",
    "| Blur | Gaussian sigma fraction: 0.002, 0.004, 0.007, 0.010, 0.015 |",
    "| Noise | Gaussian noise standard deviation: 5, 10, 20, 35, 50 |",
    "| Contrast | Contrast factors: 0.8, 0.6, 0.45, 0.3, 0.2 |",
    "| Brightness up | Intensity shifts: +15, +30, +50, +70, +90 |",
    "| Brightness down | Intensity shifts: -15, -30, -50, -70, -90 |",
    "| JPEG | Quality levels: 60, 40, 25, 15, 8 |",
    "| Downscale | Scale factors: 0.75, 0.5, 0.35, 0.25, 0.15 |",
    "",
    "This produced 36 conditions per image: one baseline and 35 corrupted "
    "conditions. The sweep contains 31,464 prediction records "
    "(874 images × 36 conditions).",
    "",
    "## 4. Baseline performance",
    "",
    "| Metric | Baseline result |",
    "|---|---:|",
    f"| Accuracy | {baseline['accuracy']} |",
    f"| Sensitivity | {baseline['sensitivity']} |",
    f"| Specificity | {baseline['specificity']} |",
    f"| ROC-AUC | {baseline['auc']} |",
    "",
    "## 5. Results across corruption severity",
    "",
    "Each entry reports the metric estimate followed by its 95% bootstrap "
    "confidence interval.",
    "",
    "| Corruption | Severity | Accuracy | Sensitivity | Specificity | ROC-AUC |",
    "|---|---:|---:|---:|---:|---:|"
]

for corruption in corruptions:
    for severity in range(1, 6):
        row = merged[
            (merged.corruption == corruption) &
            (merged.severity == severity)
        ].iloc[0]
        values = []
        for metric in metrics:
            values.append(cell({
                "value": row[f"{metric}_value"],
                "low": row[f"{metric}_low"],
                "high": row[f"{metric}_high"]
            }))
        lines.append(
            f"| {corruption} | {severity} | " + " | ".join(values) + " |"
        )

lines += [
    "",
    "## 6. Statistical analysis",
    "",
    "For each metric and corruption-severity condition, 500 bootstrap "
    "resamples were used. The 95% confidence interval was calculated "
    "using the 2.5th and 97.5th percentiles of the bootstrap estimates. "
    "The bootstrap seed was 42.",
    "",
    "Metrics:",
    "- Accuracy: proportion of correctly classified images.",
    "- Sensitivity: proportion of pneumonia cases correctly identified.",
    "- Specificity: proportion of normal cases correctly identified.",
    "- ROC-AUC: area under the receiver operating characteristic curve.",
    "",
    "## 7. Figures and supporting data",
    "",
    "- [Accuracy vs severity](analysis/fig_accuracy_vs_severity.png)",
    "- [Sensitivity vs severity](analysis/fig_sensitivity_vs_severity.png)",
    "- [Specificity vs severity](analysis/fig_specificity_vs_severity.png)",
    "- [ROC-AUC vs severity](analysis/fig_auc_vs_severity.png)",
    "- [Raw benchmark predictions](sweep.csv)",
    "",
    "Metric summary CSV files are available in the `analysis` folder.",
    "",
    "## 8. Limitations",
    "",
    "- Evaluation uses one held-out test split and one frozen model checkpoint.",
    "- The corruptions are controlled transformations and may not represent "
    "every real-world acquisition condition.",
    "- Results should not be interpreted as clinical validation or used "
    "for diagnosis.",
    "- Confidence intervals are calculated separately for each condition; "
    "they are not paired confidence intervals for differences between "
    "corruption levels.",
    "",
    "## 9. Conclusion",
    "",
    "The experiment provides a reproducible evaluation of the model under "
    "seven controlled image-quality corruptions at five severity levels. "
    "The baseline and corruption results, with bootstrap confidence "
    "intervals, provide a quantitative basis for studying model robustness.",
    "",
    "**Research and educational use only — not for clinical diagnosis.**",
    ""
]

report = root / "REAL_BENCHMARK_REPORT.md"
report.write_text("\n".join(lines), encoding="utf-8")
print("Report created:", report)
print("Corruption conditions:", len(merged))
print("Metrics included:", ", ".join(metrics))
