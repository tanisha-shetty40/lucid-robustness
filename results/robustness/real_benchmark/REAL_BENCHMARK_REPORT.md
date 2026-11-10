# LUCID — Real-Data Robustness Benchmark Report

## 1. Objective

Evaluate how controlled image-quality corruptions affect the performance of the frozen LUCID pneumonia-classification model.

## 2. Experimental setup

- Model: ResNet18, loaded from `models/classifier.pt`.
- Input size: 224 × 224 pixels.
- Classes: NORMAL and PNEUMONIA.
- Positive class: PNEUMONIA.
- Evaluation: original held-out test split, 874 images.
- Preprocessing: grayscale converted to three channels, resized to 224 × 224, converted to tensor, and normalized using ImageNet mean and standard deviation.
- Model inference: evaluation mode on CPU.
- Decision threshold: pneumonia probability >= 0.5.
- Reproducibility: stable SHA-256-derived corruption seed using image ID, corruption, severity, and base seed 42.

## 3. Corruption protocol

Each of the seven corruption types was applied at five severity levels. Severity 0 represents the unchanged baseline.

| Corruption | Severity parameters (1 to 5) |
|---|---|
| Blur | Gaussian sigma fraction: 0.002, 0.004, 0.007, 0.010, 0.015 |
| Noise | Gaussian noise standard deviation: 5, 10, 20, 35, 50 |
| Contrast | Contrast factors: 0.8, 0.6, 0.45, 0.3, 0.2 |
| Brightness up | Intensity shifts: +15, +30, +50, +70, +90 |
| Brightness down | Intensity shifts: -15, -30, -50, -70, -90 |
| JPEG | Quality levels: 60, 40, 25, 15, 8 |
| Downscale | Scale factors: 0.75, 0.5, 0.35, 0.25, 0.15 |

This produced 36 conditions per image: one baseline and 35 corrupted conditions. The sweep contains 31,464 prediction records (874 images × 36 conditions).

## 4. Baseline performance

| Metric | Baseline result |
|---|---:|
| Accuracy | 98.17% (97.31%–98.97%) |
| Sensitivity | 98.15% (97.15%–99.09%) |
| Specificity | 98.21% (96.58%–99.57%) |
| ROC-AUC | 99.79% (99.59%–99.93%) |

## 5. Results across corruption severity

Each entry reports the metric estimate followed by its 95% bootstrap confidence interval.

| Corruption | Severity | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| blur | 1 | 98.51% (97.71%–99.31%) | 99.38% (98.65%–99.85%) | 95.98% (93.24%–98.56%) | 99.76% (99.56%–99.94%) |
| blur | 2 | 95.77% (94.45%–97.14%) | 99.85% (99.41%–100.00%) | 83.93% (79.26%–88.89%) | 99.60% (99.24%–99.88%) |
| blur | 3 | 85.58% (83.52%–87.87%) | 100.00% (100.00%–100.00%) | 43.75% (37.08%–50.98%) | 99.21% (98.56%–99.71%) |
| blur | 4 | 76.77% (74.08%–79.52%) | 100.00% (100.00%–100.00%) | 9.38% (5.85%–13.15%) | 98.67% (97.83%–99.39%) |
| blur | 5 | 74.37% (71.62%–77.23%) | 100.00% (100.00%–100.00%) | 0.00% (0.00%–0.00%) | 97.05% (95.71%–98.20%) |
| brightness_down | 1 | 98.40% (97.48%–99.20%) | 98.46% (97.45%–99.38%) | 98.21% (96.44%–99.59%) | 99.76% (99.52%–99.94%) |
| brightness_down | 2 | 98.40% (97.48%–99.14%) | 98.77% (97.93%–99.54%) | 97.32% (94.91%–99.13%) | 99.69% (99.40%–99.92%) |
| brightness_down | 3 | 97.94% (96.91%–98.86%) | 98.62% (97.57%–99.39%) | 95.98% (93.46%–98.60%) | 99.61% (99.25%–99.86%) |
| brightness_down | 4 | 96.91% (95.65%–98.00%) | 97.38% (95.95%–98.49%) | 95.54% (92.72%–98.21%) | 99.50% (99.15%–99.80%) |
| brightness_down | 5 | 95.88% (94.62%–97.14%) | 97.23% (95.84%–98.46%) | 91.96% (88.58%–95.27%) | 99.21% (98.81%–99.54%) |
| brightness_up | 1 | 97.94% (96.85%–98.80%) | 97.69% (96.41%–98.73%) | 98.66% (96.91%–100.00%) | 99.80% (99.61%–99.92%) |
| brightness_up | 2 | 97.48% (96.57%–98.51%) | 97.23% (95.95%–98.47%) | 98.21% (96.45%–99.58%) | 99.79% (99.62%–99.91%) |
| brightness_up | 3 | 97.25% (96.22%–98.28%) | 97.23% (96.03%–98.44%) | 97.32% (95.28%–99.16%) | 99.73% (99.56%–99.88%) |
| brightness_up | 4 | 96.57% (95.31%–97.71%) | 96.77% (95.43%–98.10%) | 95.98% (93.19%–98.24%) | 99.58% (99.29%–99.78%) |
| brightness_up | 5 | 95.31% (93.94%–96.57%) | 94.92% (93.21%–96.45%) | 96.43% (93.89%–98.67%) | 99.46% (99.13%–99.71%) |
| contrast | 1 | 97.94% (96.91%–98.74%) | 97.69% (96.43%–98.76%) | 98.66% (96.88%–100.00%) | 99.79% (99.63%–99.92%) |
| contrast | 2 | 97.71% (96.80%–98.63%) | 97.54% (96.31%–98.63%) | 98.21% (96.36%–99.56%) | 99.76% (99.60%–99.89%) |
| contrast | 3 | 97.37% (96.34%–98.51%) | 97.54% (96.15%–98.64%) | 96.88% (94.51%–99.07%) | 99.68% (99.48%–99.85%) |
| contrast | 4 | 97.25% (96.00%–98.17%) | 98.15% (97.04%–99.09%) | 94.64% (91.33%–97.59%) | 99.39% (98.91%–99.70%) |
| contrast | 5 | 92.45% (90.73%–94.05%) | 99.38% (98.64%–99.85%) | 72.32% (66.59%–78.07%) | 98.34% (97.57%–99.06%) |
| downscale | 1 | 98.40% (97.60%–99.20%) | 98.46% (97.41%–99.24%) | 98.21% (96.52%–99.57%) | 99.79% (99.59%–99.94%) |
| downscale | 2 | 98.40% (97.60%–99.20%) | 98.46% (97.51%–99.25%) | 98.21% (96.53%–99.59%) | 99.79% (99.60%–99.95%) |
| downscale | 3 | 98.28% (97.37%–99.08%) | 98.77% (97.90%–99.54%) | 96.88% (94.06%–99.02%) | 99.79% (99.59%–99.93%) |
| downscale | 4 | 98.05% (97.08%–98.86%) | 99.08% (98.36%–99.69%) | 95.09% (91.95%–97.57%) | 99.77% (99.58%–99.91%) |
| downscale | 5 | 97.37% (96.34%–98.46%) | 99.85% (99.53%–100.00%) | 90.18% (85.99%–93.96%) | 99.64% (99.33%–99.88%) |
| jpeg | 1 | 98.28% (97.37%–99.08%) | 98.15% (97.12%–99.07%) | 98.66% (96.94%–100.00%) | 99.79% (99.56%–99.92%) |
| jpeg | 2 | 98.28% (97.42%–99.08%) | 98.31% (97.36%–99.24%) | 98.21% (96.28%–99.57%) | 99.80% (99.63%–99.94%) |
| jpeg | 3 | 98.40% (97.60%–99.08%) | 98.46% (97.47%–99.39%) | 98.21% (96.26%–99.58%) | 99.79% (99.60%–99.93%) |
| jpeg | 4 | 98.40% (97.48%–99.08%) | 98.77% (97.82%–99.53%) | 97.32% (95.28%–99.11%) | 99.80% (99.62%–99.93%) |
| jpeg | 5 | 98.05% (97.25%–98.97%) | 99.08% (98.32%–99.69%) | 95.09% (92.49%–97.85%) | 99.74% (99.54%–99.91%) |
| noise | 1 | 98.05% (97.03%–98.86%) | 98.00% (96.79%–98.92%) | 98.21% (96.29%–99.56%) | 99.78% (99.56%–99.93%) |
| noise | 2 | 98.51% (97.60%–99.26%) | 98.62% (97.64%–99.39%) | 98.21% (96.30%–99.58%) | 99.78% (99.58%–99.93%) |
| noise | 3 | 98.05% (97.14%–98.86%) | 98.92% (98.12%–99.69%) | 95.54% (92.81%–97.85%) | 99.72% (99.48%–99.90%) |
| noise | 4 | 97.37% (96.34%–98.40%) | 99.38% (98.80%–99.85%) | 91.52% (87.76%–94.94%) | 99.61% (99.29%–99.85%) |
| noise | 5 | 96.34% (94.85%–97.60%) | 99.54% (98.94%–100.00%) | 87.05% (82.06%–91.19%) | 99.43% (98.86%–99.85%) |

## 6. Statistical analysis

For each metric and corruption-severity condition, 500 bootstrap resamples were used. The 95% confidence interval was calculated using the 2.5th and 97.5th percentiles of the bootstrap estimates. The bootstrap seed was 42.

Metrics:
- Accuracy: proportion of correctly classified images.
- Sensitivity: proportion of pneumonia cases correctly identified.
- Specificity: proportion of normal cases correctly identified.
- ROC-AUC: area under the receiver operating characteristic curve.

## 7. Figures and supporting data

- [Accuracy vs severity](analysis/fig_accuracy_vs_severity.png)
- [Sensitivity vs severity](analysis/fig_sensitivity_vs_severity.png)
- [Specificity vs severity](analysis/fig_specificity_vs_severity.png)
- [ROC-AUC vs severity](analysis/fig_auc_vs_severity.png)
- [Raw benchmark predictions](sweep.csv)

Metric summary CSV files are available in the `analysis` folder.

## 8. Limitations

- Evaluation uses one held-out test split and one frozen model checkpoint.
- The corruptions are controlled transformations and may not represent every real-world acquisition condition.
- Results should not be interpreted as clinical validation or used for diagnosis.
- Confidence intervals are calculated separately for each condition; they are not paired confidence intervals for differences between corruption levels.

## 9. Conclusion

The experiment provides a reproducible evaluation of the model under seven controlled image-quality corruptions at five severity levels. The baseline and corruption results, with bootstrap confidence intervals, provide a quantitative basis for studying model robustness.

**Research and educational use only — not for clinical diagnosis.**
