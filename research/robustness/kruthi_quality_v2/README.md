\# Kruthi: Quality Score v2



\## Problem

This module explores whether five image-quality indicators can help estimate when a model prediction may be wrong.



\## Current status

The five indicators, quality score, fitting helper, and synthetic demo are implemented. Synthetic indicator tests pass. Real model evaluation is not complete.




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
- **Clipping:** The score gives up to 15 points when clipped pixels are at or below 1%, with points decreasing up to 20%.