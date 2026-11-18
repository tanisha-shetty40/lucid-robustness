\# Kruthi: Quality Score v2



\## Problem

This module explores whether five image-quality indicators can help estimate when a model prediction may be wrong.



\## Current status

The five indicators, quality score, fitting helper, and synthetic demo are implemented. Synthetic indicator tests pass. The real test-sweep comparison is complete for 874 approved test images across 31,464 conditions. The classifier was not rerun. The sweep’s seed method differs from the manual’s rule and is documented in the results.




\## Indicators

\- Histogram spread: the range between the 1st and 99th brightness percentiles.

\- Edge density: the fraction of pixels marked as edges.

\- Entropy: the variety of gray levels in the image.

\- Blockiness: how much pixel jumps differ at 8-pixel block boundaries.

\- Local contrast: average brightness variation within 16 x 16 pixel tiles.



\## Demo

The demo uses synthetic geometric shapes only. Its output is not a medical-image result and must not be reported as a research finding.



```powershell

$env:PYTHONPATH="backend;."

python -m research.robustness.kruthi\_quality\_v2.run --demo

### Existing quality score: five-image validation check

These are the first five validation entries. The weighted component points add up to the existing score, which is rounded to the nearest integer. Image files and filenames are not included.

| Example | Brightness | Contrast | Sharpness | Noise | Clipping % | Weighted points (brightness + contrast + sharpness + noise + clipping) | Score |
|---|---:|---:|---:|---:|---:|---|---:|
| 1 | 119.80 | 55.09 | 101.91 | 1.96 | 4.71 | 22.95 + 21.93 + 5.85 + 15.00 + 12.07 = 77.80 | 78 |
| 2 | 135.52 | 50.77 | 15.84 | 0.75 | 0.00 | 23.12 + 19.23 + 0.00 + 15.00 + 15.00 = 72.35 | 72 |
| 3 | 110.23 | 67.99 | 43.14 | 0.76 | 7.99 | 20.56 + 25.00 + 1.65 + 15.00 + 9.48 = 71.69 | 72 |
| 4 | 122.13 | 68.16 | 94.92 | 1.72 | 12.47 | 23.53 + 25.00 + 5.35 + 15.00 + 5.95 = 74.83 | 75 |
| 5 | 130.13 | 58.66 | 111.33 | 1.96 | 5.18 | 24.47 + 24.16 + 6.52 + 15.00 + 11.70 = 81.85 | 82 |

- **Brightness:** The score gives up to 25 points when mean intensity is near 128 and fewer points as it moves away.
- **Contrast:** The score gives up to 25 points as grayscale standard deviation rises from 20 to 60.
- **Sharpness:** The score gives up to 20 points as Laplacian variance rises from 20 to 300.
- **Noise:** The score gives up to 15 points when estimated noise is low, with points decreasing as it rises from 2 to 12.
- **Clipping:** The score gives up to 15 points when clipped pixels are at or below 1%, with points decreasing up to 20%.# Quality Score V2 test comparison

Research and educational use only; not a diagnostic result.

Approved test images: 874. Rows analyzed: 31464. The classifier was not rerun.

The supplied sweep used condition-specific hash seeds, which differs from the manual's `42 + image index` rule. This deviation is recorded for review.

| Corruption | Rows | Old AUC (95% CI) | V2 AUC (95% CI) | V2 - old |
|---|---:|---:|---:|---:|
| overall | 31464 | 0.553 (0.537–0.566) | 0.713 (0.696–0.728) | 0.160 |
| blur | 4370 | 0.405 (0.381–0.428) | 0.843 (0.826–0.859) | 0.438 |
| noise | 4370 | 0.538 (0.478–0.597) | 0.507 (0.453–0.557) | -0.031 |
| contrast | 4370 | 0.633 (0.605–0.657) | 0.757 (0.728–0.786) | 0.124 |
| brightness_up | 4370 | 0.542 (0.500–0.590) | 0.533 (0.488–0.576) | -0.009 |
| brightness_down | 4370 | 0.596 (0.543–0.647) | 0.459 (0.424–0.498) | -0.137 |
| jpeg | 4370 | 0.503 (0.443–0.562) | 0.515 (0.468–0.563) | 0.012 |
| downscale | 4370 | 0.552 (0.494–0.602) | 0.517 (0.461–0.574) | -0.035 |

## Validation Fitting & Summary
Weights were fitted via logistic regression on validation images only (seeds >= 1000)  to prevent test data leakage.Overall AUC improved by +0.160 (0.713 vs 0.553) due to blur (+0.438) and contrast (+0.124), while it did not help on brightness_down (-0.137), downscale (-0.035), noise (-0.031), brightness_up (-0.009), or jpeg (+0.012).
