# LUCID — Real-Data Robustness and Image-Quality Analysis Report

## 1. Objective

Evaluate how controlled image-quality corruptions affect the performance of the frozen LUCID pneumonia-classification model, and investigate associations between image-quality measurements and prediction errors.

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

## 7. Image-quality analysis

Image-quality measurements were calculated on the original grayscale images before model resizing and normalization. Each corrupted image was compared with its corresponding original image.

The following metrics were calculated:
- PSNR: peak signal-to-noise ratio, measuring pixel-level distortion.
- SSIM: structural similarity index, measuring structural similarity.
- MSE: mean squared pixel error.

The quality measurements were joined to the corresponding model predictions using image ID, corruption type, and severity. Prediction error rate is the percentage of images incorrectly classified in each condition.

| Corruption | Severity | Mean PSNR (dB) | Mean SSIM | Mean MSE | Errors | Error rate |
|---|---:|---:|---:|---:|---:|---:|
| blur | 1 | 36.409 | 0.9205 | 18.6609 | 13/874 | 1.49% |
| blur | 2 | 33.165 | 0.8751 | 36.0414 | 37/874 | 4.23% |
| blur | 3 | 30.936 | 0.8406 | 58.8924 | 126/874 | 14.42% |
| blur | 4 | 29.615 | 0.8225 | 79.4280 | 203/874 | 23.23% |
| blur | 5 | 28.026 | 0.8038 | 114.5868 | 224/874 | 25.63% |
| brightness_down | 1 | 24.928 | 0.9526 | 209.5267 | 14/874 | 1.60% |
| brightness_down | 2 | 19.003 | 0.8880 | 819.8641 | 14/874 | 1.60% |
| brightness_down | 3 | 14.692 | 0.7792 | 2213.0004 | 18/874 | 2.06% |
| brightness_down | 4 | 11.933 | 0.6520 | 4180.0032 | 27/874 | 3.09% |
| brightness_down | 5 | 9.967 | 0.5203 | 6583.4510 | 36/874 | 4.12% |
| brightness_up | 1 | 24.614 | 0.9298 | 224.7530 | 18/874 | 2.06% |
| brightness_up | 2 | 18.602 | 0.8996 | 897.1539 | 22/874 | 2.52% |
| brightness_up | 3 | 14.223 | 0.8546 | 2459.9942 | 24/874 | 2.75% |
| brightness_up | 4 | 11.489 | 0.7997 | 4624.1710 | 30/874 | 3.43% |
| brightness_up | 5 | 9.666 | 0.7413 | 7057.5222 | 41/874 | 4.69% |
| contrast | 1 | 27.098 | 0.9208 | 134.7531 | 18/874 | 2.06% |
| contrast | 2 | 21.080 | 0.8779 | 538.7086 | 20/874 | 2.29% |
| contrast | 3 | 18.313 | 0.8352 | 1018.8080 | 23/874 | 2.63% |
| contrast | 4 | 16.219 | 0.7822 | 1650.0598 | 24/874 | 2.75% |
| contrast | 5 | 15.059 | 0.7414 | 2155.1744 | 66/874 | 7.55% |
| downscale | 1 | 40.960 | 0.9768 | 6.2722 | 14/874 | 1.60% |
| downscale | 2 | 38.489 | 0.9552 | 10.8521 | 14/874 | 1.60% |
| downscale | 3 | 36.648 | 0.9305 | 16.1282 | 15/874 | 1.72% |
| downscale | 4 | 35.237 | 0.9083 | 21.8614 | 17/874 | 1.95% |
| downscale | 5 | 33.055 | 0.8722 | 35.1158 | 23/874 | 2.63% |
| jpeg | 1 | 43.127 | 0.9680 | 3.7156 | 15/874 | 1.72% |
| jpeg | 2 | 39.491 | 0.9245 | 7.8205 | 15/874 | 1.72% |
| jpeg | 3 | 37.760 | 0.9076 | 11.4519 | 14/874 | 1.60% |
| jpeg | 4 | 35.876 | 0.8558 | 17.4061 | 14/874 | 1.60% |
| jpeg | 5 | 33.122 | 0.7907 | 32.2489 | 17/874 | 1.95% |
| noise | 1 | 34.273 | 0.7735 | 24.3218 | 17/874 | 1.95% |
| noise | 2 | 28.287 | 0.4793 | 96.5109 | 13/874 | 1.49% |
| noise | 3 | 22.330 | 0.2035 | 380.4468 | 17/874 | 1.95% |
| noise | 4 | 17.641 | 0.0848 | 1120.0811 | 23/874 | 2.63% |
| noise | 5 | 14.831 | 0.0475 | 2139.4492 | 32/874 | 3.66% |

For mean PSNR, infinite and non-finite individual values are excluded from the condition-level average. Infinite PSNR occurs when an individual image has zero MSE, meaning no pixel difference was measured. If all values in a condition are non-finite, the mean is reported as Not available.

## 8. Correlation analysis and multiple-testing correction

Spearman rank correlations were calculated between each image-quality metric and the binary prediction-error outcome separately for every corruption-severity condition. Each test uses the 874 test images.

The analysis includes 105 tests: seven corruptions × five severity levels × three image-quality metrics.

Benjamini–Hochberg false discovery rate correction was applied across the 105 p-values. A corrected q-value below 0.05 was treated as statistically significant.

- Total correlation tests: 105.
- Tests significant after FDR correction: 55.
- FDR significance threshold: q < 0.05.

| Corruption | Severity | Metric | Spearman rho | p-value | q-value | FDR significant |
|---|---:|---|---:|---:|---:|---|
| blur | 1 | mse | -0.0028 | 0.933 | 0.94 | No |
| blur | 1 | psnr | 0.0028 | 0.933 | 0.94 | No |
| blur | 1 | ssim | -0.0416 | 0.219 | 0.302 | No |
| blur | 2 | mse | 0.0717 | 0.0341 | 0.0627 | No |
| blur | 2 | psnr | -0.0717 | 0.0341 | 0.0627 | No |
| blur | 2 | ssim | -0.0938 | 0.00553 | 0.0153 | Yes |
| blur | 3 | mse | 0.2946 | 5.92e-19 | 6.9e-18 | Yes |
| blur | 3 | psnr | -0.2946 | 5.92e-19 | 6.9e-18 | Yes |
| blur | 3 | ssim | -0.3333 | 4.1e-24 | 6.15e-23 | Yes |
| blur | 4 | mse | 0.4446 | 1.22e-43 | 2.13e-42 | Yes |
| blur | 4 | psnr | -0.4446 | 1.22e-43 | 2.13e-42 | Yes |
| blur | 4 | ssim | -0.4783 | 3.62e-51 | 9.49e-50 | Yes |
| blur | 5 | mse | 0.5303 | 1.54e-64 | 8.1e-63 | Yes |
| blur | 5 | psnr | -0.5303 | 1.54e-64 | 8.1e-63 | Yes |
| blur | 5 | ssim | -0.5279 | 7.21e-64 | 2.52e-62 | Yes |
| brightness_down | 1 | mse | -0.0790 | 0.0194 | 0.0385 | Yes |
| brightness_down | 1 | psnr | 0.0790 | 0.0194 | 0.0385 | Yes |
| brightness_down | 1 | ssim | 0.0046 | 0.893 | 0.919 | No |
| brightness_down | 2 | mse | -0.0968 | 0.00419 | 0.0119 | Yes |
| brightness_down | 2 | psnr | 0.0968 | 0.00419 | 0.0119 | Yes |
| brightness_down | 2 | ssim | -0.0153 | 0.652 | 0.699 | No |
| brightness_down | 3 | mse | -0.0674 | 0.0465 | 0.08 | No |
| brightness_down | 3 | psnr | 0.0674 | 0.0465 | 0.08 | No |
| brightness_down | 3 | ssim | -0.0353 | 0.297 | 0.38 | No |
| brightness_down | 4 | mse | -0.1152 | 0.000648 | 0.00425 | Yes |
| brightness_down | 4 | psnr | 0.1152 | 0.000648 | 0.00425 | Yes |
| brightness_down | 4 | ssim | -0.0342 | 0.312 | 0.39 | No |
| brightness_down | 5 | mse | -0.1121 | 0.000902 | 0.00512 | Yes |
| brightness_down | 5 | psnr | 0.1121 | 0.000902 | 0.00512 | Yes |
| brightness_down | 5 | ssim | -0.0553 | 0.103 | 0.158 | No |
| brightness_up | 1 | mse | -0.0271 | 0.424 | 0.5 | No |
| brightness_up | 1 | psnr | 0.0271 | 0.424 | 0.5 | No |
| brightness_up | 1 | ssim | -0.0970 | 0.0041 | 0.0119 | Yes |
| brightness_up | 2 | mse | -0.0768 | 0.0232 | 0.0443 | Yes |
| brightness_up | 2 | psnr | 0.0768 | 0.0232 | 0.0443 | Yes |
| brightness_up | 2 | ssim | -0.1015 | 0.00267 | 0.0108 | Yes |
| brightness_up | 3 | mse | -0.0874 | 0.00971 | 0.0232 | Yes |
| brightness_up | 3 | psnr | 0.0874 | 0.00971 | 0.0232 | Yes |
| brightness_up | 3 | ssim | -0.0978 | 0.00382 | 0.0118 | Yes |
| brightness_up | 4 | mse | -0.1216 | 0.000315 | 0.00236 | Yes |
| brightness_up | 4 | psnr | 0.1216 | 0.000315 | 0.00236 | Yes |
| brightness_up | 4 | ssim | -0.0839 | 0.013 | 0.0285 | Yes |
| brightness_up | 5 | mse | -0.1248 | 0.000218 | 0.0019 | Yes |
| brightness_up | 5 | psnr | 0.1248 | 0.000218 | 0.0019 | Yes |
| brightness_up | 5 | ssim | -0.0866 | 0.0104 | 0.0243 | Yes |
| contrast | 1 | mse | 0.1061 | 0.00168 | 0.00768 | Yes |
| contrast | 1 | psnr | -0.1061 | 0.00168 | 0.00768 | Yes |
| contrast | 1 | ssim | -0.1003 | 0.003 | 0.0109 | Yes |
| contrast | 2 | mse | 0.1114 | 0.000974 | 0.00512 | Yes |
| contrast | 2 | psnr | -0.1114 | 0.000974 | 0.00512 | Yes |
| contrast | 2 | ssim | -0.0998 | 0.00315 | 0.011 | Yes |
| contrast | 3 | mse | 0.1038 | 0.00213 | 0.00895 | Yes |
| contrast | 3 | psnr | -0.1038 | 0.00213 | 0.00895 | Yes |
| contrast | 3 | ssim | -0.0978 | 0.0038 | 0.0118 | Yes |
| contrast | 4 | mse | 0.0814 | 0.0161 | 0.0332 | Yes |
| contrast | 4 | psnr | -0.0814 | 0.0161 | 0.0332 | Yes |
| contrast | 4 | ssim | -0.1079 | 0.00139 | 0.00697 | Yes |
| contrast | 5 | mse | 0.0984 | 0.0036 | 0.0118 | Yes |
| contrast | 5 | psnr | -0.0984 | 0.0036 | 0.0118 | Yes |
| contrast | 5 | ssim | -0.1263 | 0.000181 | 0.0019 | Yes |
| downscale | 1 | mse | -0.0630 | 0.0628 | 0.101 | No |
| downscale | 1 | psnr | 0.0630 | 0.0628 | 0.101 | No |
| downscale | 1 | ssim | 0.0272 | 0.423 | 0.5 | No |
| downscale | 2 | mse | -0.0635 | 0.0604 | 0.101 | No |
| downscale | 2 | psnr | 0.0635 | 0.0604 | 0.101 | No |
| downscale | 2 | ssim | 0.0402 | 0.235 | 0.32 | No |
| downscale | 3 | mse | -0.0386 | 0.255 | 0.334 | No |
| downscale | 3 | psnr | 0.0386 | 0.255 | 0.334 | No |
| downscale | 3 | ssim | -0.0026 | 0.94 | 0.94 | No |
| downscale | 4 | mse | 0.0161 | 0.634 | 0.693 | No |
| downscale | 4 | psnr | -0.0161 | 0.634 | 0.693 | No |
| downscale | 4 | ssim | -0.0445 | 0.189 | 0.284 | No |
| downscale | 5 | mse | 0.0257 | 0.448 | 0.517 | No |
| downscale | 5 | psnr | -0.0257 | 0.448 | 0.517 | No |
| downscale | 5 | ssim | -0.0244 | 0.472 | 0.538 | No |
| jpeg | 1 | mse | -0.0573 | 0.0906 | 0.142 | No |
| jpeg | 1 | psnr | 0.0573 | 0.0906 | 0.142 | No |
| jpeg | 1 | ssim | 0.0396 | 0.242 | 0.326 | No |
| jpeg | 2 | mse | -0.0437 | 0.197 | 0.284 | No |
| jpeg | 2 | psnr | 0.0437 | 0.197 | 0.284 | No |
| jpeg | 2 | ssim | 0.0152 | 0.653 | 0.699 | No |
| jpeg | 3 | mse | -0.0304 | 0.369 | 0.451 | No |
| jpeg | 3 | psnr | 0.0304 | 0.369 | 0.451 | No |
| jpeg | 3 | ssim | 0.0125 | 0.712 | 0.755 | No |
| jpeg | 4 | mse | -0.0066 | 0.844 | 0.878 | No |
| jpeg | 4 | psnr | 0.0066 | 0.844 | 0.878 | No |
| jpeg | 4 | ssim | -0.0418 | 0.217 | 0.302 | No |
| jpeg | 5 | mse | 0.0191 | 0.572 | 0.639 | No |
| jpeg | 5 | psnr | -0.0191 | 0.572 | 0.639 | No |
| jpeg | 5 | ssim | -0.0829 | 0.0143 | 0.0306 | Yes |
| noise | 1 | mse | -0.0930 | 0.00591 | 0.0155 | Yes |
| noise | 1 | psnr | 0.0930 | 0.00591 | 0.0155 | Yes |
| noise | 1 | ssim | -0.0369 | 0.276 | 0.358 | No |
| noise | 2 | mse | -0.1007 | 0.00288 | 0.0108 | Yes |
| noise | 2 | psnr | 0.1007 | 0.00288 | 0.0108 | Yes |
| noise | 2 | ssim | -0.0351 | 0.3 | 0.38 | No |
| noise | 3 | mse | -0.0689 | 0.0417 | 0.0742 | No |
| noise | 3 | psnr | 0.0689 | 0.0417 | 0.0742 | No |
| noise | 3 | ssim | -0.0437 | 0.197 | 0.284 | No |
| noise | 4 | mse | -0.0888 | 0.00862 | 0.0216 | Yes |
| noise | 4 | psnr | 0.0888 | 0.00862 | 0.0216 | Yes |
| noise | 4 | ssim | -0.0434 | 0.2 | 0.284 | No |
| noise | 5 | mse | -0.0852 | 0.0117 | 0.0262 | Yes |
| noise | 5 | psnr | 0.0852 | 0.0117 | 0.0262 | Yes |
| noise | 5 | ssim | -0.0440 | 0.194 | 0.284 | No |

### Strongest FDR-significant associations

| Corruption | Severity | Metric | Spearman rho | q-value |
|---|---:|---|---:|---:|
| blur | 5 | psnr | -0.5303 | 8.1e-63 |
| blur | 5 | mse | 0.5303 | 8.1e-63 |
| blur | 5 | ssim | -0.5279 | 2.52e-62 |
| blur | 4 | ssim | -0.4783 | 9.49e-50 |
| blur | 4 | mse | 0.4446 | 2.13e-42 |
| blur | 4 | psnr | -0.4446 | 2.13e-42 |
| blur | 3 | ssim | -0.3333 | 6.15e-23 |
| blur | 3 | psnr | -0.2946 | 6.9e-18 |
| blur | 3 | mse | 0.2946 | 6.9e-18 |
| contrast | 5 | ssim | -0.1263 | 0.0019 |

These are associations, not evidence that a particular quality metric causes classification errors. The 105 tests are related because they reuse the same test images and correlated image-quality measurements. PSNR and MSE are mathematically related and should not be treated as independent evidence.

## 9. Key findings

- Blur produced the largest observed increase in prediction errors. At severity 5, the error rate was approximately 25.63%, compared with 1.83% at baseline.
- Blur severity 3 and above showed a marked increase in error rate.
- Contrast severity 5 also increased the error rate, to approximately 7.55%.
- Brightness changes produced more moderate increases in errors.
- Noise caused a substantial reduction in mean SSIM at high severity, but this did not translate into a proportionate increase in prediction errors. This illustrates that image similarity metrics alone do not fully predict classifier performance.
- JPEG compression and downscaling produced comparatively smaller increases in prediction errors in this experiment.
- Statistical significance does not necessarily indicate a large or practically important association.

## 10. Figures and supporting data

- [Accuracy vs severity](analysis/fig_accuracy_vs_severity.png)
- [Sensitivity vs severity](analysis/fig_sensitivity_vs_severity.png)
- [Specificity vs severity](analysis/fig_specificity_vs_severity.png)
- [ROC-AUC vs severity](analysis/fig_auc_vs_severity.png)
- [Error rate vs severity](error_rate_vs_severity.png)
- [SSIM vs error rate](ssim_vs_error_rate.png)
- [Raw benchmark predictions](sweep.csv)
- [Image-quality and prediction analysis](quality_prediction_analysis.csv)
- [Condition-level quality summary](quality_error_condition_summary.csv)
- [Spearman correlation results](quality_error_correlations.csv)
- [FDR-corrected correlation results](quality_error_correlations_corrected.csv)

Metric summary CSV files are available in the `analysis` folder.

## 11. Limitations

- Evaluation uses one held-out test split and one frozen model checkpoint.
- The corruptions are controlled transformations and may not represent every real-world acquisition condition.
- Results should not be interpreted as clinical validation or used for diagnosis.
- Confidence intervals are calculated separately for each condition; they are not paired confidence intervals for differences between corruption levels.
- Correlation analysis is exploratory and does not establish causation.
- The 105 correlation tests are statistically related because they reuse the same images and related metrics. FDR correction does not remove all dependence between tests.
- PSNR and MSE are mathematically related.
- The findings may not generalize to other datasets, scanners, acquisition settings, or model architectures.

## 12. Conclusion

The experiment provides a reproducible evaluation of the frozen model under seven controlled image-quality corruptions at five severity levels. The results show that the effect of image degradation depends on the corruption type. Blur had the largest observed impact on classification errors, while some severe changes in image similarity did not produce comparable increases in errors.

The image-quality and correlation analyses provide additional exploratory evidence about the relationship between image degradation and model prediction failures. These results are limited to this test split and checkpoint.

**Research and educational use only — not for clinical diagnosis.**
