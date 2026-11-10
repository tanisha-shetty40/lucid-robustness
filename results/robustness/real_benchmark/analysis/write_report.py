
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

analysis = Path(__file__).resolve().parent
root = analysis.parent

metrics = ["accuracy", "sensitivity", "specificity", "auc"]

corruptions = [
    "blur",
    "brightness_down",
    "brightness_up",
    "contrast",
    "downscale",
    "jpeg",
    "noise",
]

expected_conditions = 36
expected_corrupted_conditions = 35
expected_images = 874
expected_correlation_tests = 105


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def validate_columns(df, required, filename):
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename}: missing columns {sorted(missing)}"
        )


def parse_boolean(series):
    """Safely convert CSV Boolean values to Boolean dtype."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = series.astype(str).str.strip().str.lower().map(mapping)

    if converted.isna().any():
        raise ValueError(
            f"Invalid Boolean values found in column {series.name}"
        )

    return converted


def format_metric(row):
    return (
        f"{row['value']:.2%} "
        f"({row['low']:.2%}–{row['high']:.2%})"
    )


# --------------------------------------------------
# LOAD AND VALIDATE MODEL METRIC SUMMARIES
# --------------------------------------------------

data = {}

for metric in metrics:

    path = analysis / f"{metric}_summary.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    validate_columns(
        df,
        ["corruption", "severity", "value", "low", "high"],
        path.name,
    )

    if len(df) != expected_conditions:
        raise ValueError(
            f"{metric}: expected {expected_conditions} rows, "
            f"found {len(df)}"
        )

    if df[["value", "low", "high"]].isna().any().any():
        raise ValueError(
            f"{metric}: missing metric or confidence interval values"
        )

    if df.duplicated(["corruption", "severity"]).any():
        raise ValueError(
            f"{metric}: duplicate corruption/severity conditions"
        )

    data[metric] = df


# --------------------------------------------------
# BASELINE METRICS
# --------------------------------------------------

baseline = {}

for metric, df in data.items():

    selected = df[
        (df["corruption"] == "none") &
        (df["severity"] == 0)
    ]

    if len(selected) != 1:
        raise ValueError(
            f"{metric}: expected exactly one baseline condition"
        )

    baseline[metric] = format_metric(selected.iloc[0])


# --------------------------------------------------
# MERGE CORRUPTION METRICS
# --------------------------------------------------

merged = None

for metric, df in data.items():

    part = df[df["corruption"] != "none"][
        ["corruption", "severity", "value", "low", "high"]
    ].copy()

    part = part.rename(
        columns={
            "value": f"{metric}_value",
            "low": f"{metric}_low",
            "high": f"{metric}_high",
        }
    )

    merged = (
        part
        if merged is None
        else merged.merge(
            part,
            on=["corruption", "severity"],
            validate="one_to_one",
        )
    )


if len(merged) != expected_corrupted_conditions:
    raise ValueError(
        "Expected exactly 35 merged corrupted conditions"
    )


# --------------------------------------------------
# LOAD IMAGE-QUALITY ANALYSIS
# --------------------------------------------------

quality_path = root / "quality_error_condition_summary.csv"
quality_detail_path = root / "quality_prediction_analysis.csv"
correlation_path = root / "quality_error_correlations_corrected.csv"

for path in [
    quality_path,
    quality_detail_path,
    correlation_path,
]:
    if not path.exists():
        raise FileNotFoundError(path)


quality = pd.read_csv(quality_path)
quality_detail = pd.read_csv(quality_detail_path)
correlations = pd.read_csv(correlation_path)


validate_columns(
    quality,
    [
        "corruption",
        "severity",
        "mean_psnr",
        "mean_ssim",
        "mean_mse",
        "prediction_errors",
        "total_images",
        "error_rate_percent",
    ],
    quality_path.name,
)

validate_columns(
    quality_detail,
    [
        "corruption",
        "severity",
        "psnr",
    ],
    quality_detail_path.name,
)

validate_columns(
    correlations,
    [
        "corruption",
        "severity",
        "metric",
        "spearman_rho",
        "p_value",
        "sample_count",
        "q_value",
        "significant_fdr_05",
    ],
    correlation_path.name,
)


# --------------------------------------------------
# VALIDATE IMAGE-QUALITY CONDITIONS
# --------------------------------------------------

if len(quality) != expected_conditions:
    raise ValueError(
        f"Expected 36 quality conditions, found {len(quality)}"
    )

if quality.duplicated(["corruption", "severity"]).any():
    raise ValueError("Duplicate image-quality conditions found")

required_quality_values = [
    "mean_ssim",
    "mean_mse",
    "prediction_errors",
    "total_images",
    "error_rate_percent",
]

if quality[required_quality_values].isna().any().any():
    raise ValueError(
        "Missing values in image-quality summary"
    )

if len(quality_detail) != 31464:
    raise ValueError(
        f"Expected 31464 image-quality records, "
        f"found {len(quality_detail)}"
    )

if quality_detail[
    ["corruption", "severity", "psnr"]
].isna().any().any():
    raise ValueError(
        "Missing values in image-quality detail records"
    )


# --------------------------------------------------
# CORRECT MEAN PSNR CALCULATION
# --------------------------------------------------

# Infinite PSNR occurs when MSE = 0.
# Such values must not be included in a finite mean.

quality_detail["psnr"] = pd.to_numeric(
    quality_detail["psnr"],
    errors="coerce",
)

quality_detail["psnr"] = quality_detail["psnr"].replace(
    [np.inf, -np.inf],
    np.nan,
)

finite_psnr = (
    quality_detail
    .groupby(
        ["corruption", "severity"],
        as_index=False,
    )["psnr"]
    .mean()
    .rename(columns={"psnr": "finite_mean_psnr"})
)

quality = quality.merge(
    finite_psnr,
    on=["corruption", "severity"],
    how="left",
    validate="one_to_one",
)


# --------------------------------------------------
# VALIDATE CORRELATION RESULTS
# --------------------------------------------------

if len(correlations) != expected_correlation_tests:
    raise ValueError(
        f"Expected 105 correlation tests, "
        f"found {len(correlations)}"
    )

if correlations.duplicated(
    ["corruption", "severity", "metric"]
).any():
    raise ValueError("Duplicate correlation tests found")

if correlations[
    ["spearman_rho", "p_value", "q_value", "sample_count"]
].isna().any().any():
    raise ValueError(
        "Missing values in correlation results"
    )

correlations["significant_fdr_05"] = parse_boolean(
    correlations["significant_fdr_05"]
)


# --------------------------------------------------
# PREPARE QUALITY RESULTS
# --------------------------------------------------

quality_corruptions = quality[
    quality["corruption"] != "none"
].copy()

if len(quality_corruptions) != expected_corrupted_conditions:
    raise ValueError(
        "Expected 35 corrupted image-quality conditions"
    )

significant_count = int(
    correlations["significant_fdr_05"].sum()
)

significant = correlations[
    correlations["significant_fdr_05"]
].copy()

significant["absolute_rho"] = (
    significant["spearman_rho"].abs()
)

significant = significant.sort_values(
    "absolute_rho",
    ascending=False,
)


# --------------------------------------------------
# REPORT CONTENT
# --------------------------------------------------

lines = [

    "# LUCID — Real-Data Robustness and Image-Quality Analysis Report",

    "",

    "## 1. Objective",

    "",

    "Evaluate how controlled image-quality corruptions affect the "
    "performance of the frozen LUCID pneumonia-classification model, "
    "and investigate associations between image-quality measurements "
    "and prediction errors.",

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

    "Each of the seven corruption types was applied at five severity "
    "levels. Severity 0 represents the unchanged baseline.",

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

    "Each entry reports the metric estimate followed by its 95% "
    "bootstrap confidence interval.",

    "",

    "| Corruption | Severity | Accuracy | Sensitivity | Specificity | ROC-AUC |",
    "|---|---:|---:|---:|---:|---:|",

]


for corruption in corruptions:

    for severity in range(1, 6):

        selected = merged[
            (merged["corruption"] == corruption) &
            (merged["severity"] == severity)
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Missing or duplicate metric condition: "
                f"{corruption}, severity {severity}"
            )

        row = selected.iloc[0]

        values = []

        for metric in metrics:

            values.append(
                format_metric({
                    "value": row[f"{metric}_value"],
                    "low": row[f"{metric}_low"],
                    "high": row[f"{metric}_high"],
                })
            )

        lines.append(
            f"| {corruption} | {severity} | "
            + " | ".join(values)
            + " |"
        )


# --------------------------------------------------
# STATISTICAL ANALYSIS
# --------------------------------------------------

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

    "## 7. Image-quality analysis",

    "",

    "Image-quality measurements were calculated on the original grayscale "
    "images before model resizing and normalization. Each corrupted image "
    "was compared with its corresponding original image.",

    "",

    "The following metrics were calculated:",

    "- PSNR: peak signal-to-noise ratio, measuring pixel-level distortion.",
    "- SSIM: structural similarity index, measuring structural similarity.",
    "- MSE: mean squared pixel error.",

    "",

    "The quality measurements were joined to the corresponding model "
    "predictions using image ID, corruption type, and severity. "
    "Prediction error rate is the percentage of images incorrectly "
    "classified in each condition.",

    "",

    "| Corruption | Severity | Mean PSNR (dB) | Mean SSIM | Mean MSE | Errors | Error rate |",
    "|---|---:|---:|---:|---:|---:|---:|",

]


for _, row in quality_corruptions.sort_values(
    ["corruption", "severity"]
).iterrows():

    psnr_value = row["finite_mean_psnr"]

    if pd.notna(psnr_value) and np.isfinite(psnr_value):
        psnr_text = f"{psnr_value:.3f}"
    else:
        psnr_text = "Not available"

    lines.append(
        f"| {row['corruption']} | {int(row['severity'])} | "
        f"{psnr_text} | "
        f"{row['mean_ssim']:.4f} | "
        f"{row['mean_mse']:.4f} | "
        f"{int(row['prediction_errors'])}/"
        f"{int(row['total_images'])} | "
        f"{row['error_rate_percent']:.2f}% |"
    )


lines += [

    "",

    "For mean PSNR, infinite and non-finite individual values are "
    "excluded from the condition-level average. Infinite PSNR occurs "
    "when an individual image has zero MSE, meaning no pixel difference "
    "was measured. If all values in a condition are non-finite, "
    "the mean is reported as Not available.",

    "",

    "## 8. Correlation analysis and multiple-testing correction",

    "",

    "Spearman rank correlations were calculated between each image-quality "
    "metric and the binary prediction-error outcome separately for every "
    "corruption-severity condition. Each test uses the 874 test images.",

    "",

    "The analysis includes 105 tests: seven corruptions × five severity "
    "levels × three image-quality metrics.",

    "",

    "Benjamini–Hochberg false discovery rate correction was applied across "
    "the 105 p-values. A corrected q-value below 0.05 was treated as "
    "statistically significant.",

    "",

    f"- Total correlation tests: {len(correlations)}.",
    f"- Tests significant after FDR correction: {significant_count}.",
    "- FDR significance threshold: q < 0.05.",

    "",

    "| Corruption | Severity | Metric | Spearman rho | p-value | q-value | FDR significant |",
    "|---|---:|---|---:|---:|---:|---|",

]


for _, row in correlations.sort_values(
    ["corruption", "severity", "metric"]
).iterrows():

    lines.append(
        f"| {row['corruption']} | {int(row['severity'])} | "
        f"{row['metric']} | "
        f"{row['spearman_rho']:.4f} | "
        f"{row['p_value']:.3g} | "
        f"{row['q_value']:.3g} | "
        f"{'Yes' if row['significant_fdr_05'] else 'No'} |"
    )


lines += [

    "",

    "### Strongest FDR-significant associations",

    "",

    "| Corruption | Severity | Metric | Spearman rho | q-value |",
    "|---|---:|---|---:|---:|",

]


for _, row in significant.head(10).iterrows():

    lines.append(
        f"| {row['corruption']} | {int(row['severity'])} | "
        f"{row['metric']} | "
        f"{row['spearman_rho']:.4f} | "
        f"{row['q_value']:.3g} |"
    )


lines += [

    "",

    "These are associations, not evidence that a particular quality metric "
    "causes classification errors. The 105 tests are related because they "
    "reuse the same test images and correlated image-quality measurements. "
    "PSNR and MSE are mathematically related and should not be treated as "
    "independent evidence.",

    "",

    "## 9. Key findings",

    "",

    "- Blur produced the largest observed increase in prediction errors. "
    "At severity 5, the error rate was approximately 25.63%, compared with "
    "1.83% at baseline.",

    "- Blur severity 3 and above showed a marked increase in error rate.",

    "- Contrast severity 5 also increased the error rate, to approximately "
    "7.55%.",

    "- Brightness changes produced more moderate increases in errors.",

    "- Noise caused a substantial reduction in mean SSIM at high severity, "
    "but this did not translate into a proportionate increase in prediction "
    "errors. This illustrates that image similarity metrics alone do not "
    "fully predict classifier performance.",

    "- JPEG compression and downscaling produced comparatively smaller "
    "increases in prediction errors in this experiment.",

    "- Statistical significance does not necessarily indicate a large or "
    "practically important association.",

    "",

    "## 10. Figures and supporting data",

    "",

    "- [Accuracy vs severity](analysis/fig_accuracy_vs_severity.png)",
    "- [Sensitivity vs severity](analysis/fig_sensitivity_vs_severity.png)",
    "- [Specificity vs severity](analysis/fig_specificity_vs_severity.png)",
    "- [ROC-AUC vs severity](analysis/fig_auc_vs_severity.png)",
    "- [Error rate vs severity](error_rate_vs_severity.png)",
    "- [SSIM vs error rate](ssim_vs_error_rate.png)",
    "- [Raw benchmark predictions](sweep.csv)",
    "- [Image-quality and prediction analysis](quality_prediction_analysis.csv)",
    "- [Condition-level quality summary](quality_error_condition_summary.csv)",
    "- [Spearman correlation results](quality_error_correlations.csv)",
    "- [FDR-corrected correlation results](quality_error_correlations_corrected.csv)",

    "",

    "Metric summary CSV files are available in the `analysis` folder.",

    "",

    "## 11. Limitations",

    "",

    "- Evaluation uses one held-out test split and one frozen model checkpoint.",
    "- The corruptions are controlled transformations and may not represent "
    "every real-world acquisition condition.",
    "- Results should not be interpreted as clinical validation or used "
    "for diagnosis.",
    "- Confidence intervals are calculated separately for each condition; "
    "they are not paired confidence intervals for differences between "
    "corruption levels.",
    "- Correlation analysis is exploratory and does not establish causation.",
    "- The 105 correlation tests are statistically related because they "
    "reuse the same images and related metrics. FDR correction does not "
    "remove all dependence between tests.",
    "- PSNR and MSE are mathematically related.",
    "- The findings may not generalize to other datasets, scanners, "
    "acquisition settings, or model architectures.",

    "",

    "## 12. Conclusion",

    "",

    "The experiment provides a reproducible evaluation of the frozen model "
    "under seven controlled image-quality corruptions at five severity "
    "levels. The results show that the effect of image degradation depends "
    "on the corruption type. Blur had the largest observed impact on "
    "classification errors, while some severe changes in image similarity "
    "did not produce comparable increases in errors.",

    "",

    "The image-quality and correlation analyses provide additional "
    "exploratory evidence about the relationship between image degradation "
    "and model prediction failures. These results are limited to this "
    "test split and checkpoint.",

    "",

    "**Research and educational use only — not for clinical diagnosis.**",

    "",
]


# --------------------------------------------------
# WRITE REPORT
# --------------------------------------------------

report = root / "REAL_BENCHMARK_REPORT.md"

report.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

print("Report created:", report)
print("Corruption conditions:", len(merged))
print("Quality conditions:", len(quality_corruptions))
print("Image-quality records:", len(quality_detail))
print("Correlation tests:", len(correlations))
print("FDR-significant tests:", significant_count)
print("Metrics included:", ", ".join(metrics))
