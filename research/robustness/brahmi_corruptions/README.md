# Brahmi's Image Corruption Library

## Purpose

This library applies seven controlled corruptions to grayscale images to study how image quality changes affect model performance. It is intended for research and educational use, not clinical diagnosis.

## Input and output

- Input: a 2D NumPy grayscale image with dtype `uint8`.
- Output: a new `uint8` image with the same shape.
- Severity 0 returns an unchanged copy.
- Severity levels range from 1 to 5.
- Input images are not modified in place.
- Noise is reproducible when the same random seed is used.

## Corruption parameters

| Corruption | Severity 1 | Severity 2 | Severity 3 | Severity 4 | Severity 5 |
|---|---:|---:|---:|---:|---:|
| Blur (sigma fraction of shorter side) | 0.002 | 0.004 | 0.007 | 0.010 | 0.015 |
| Gaussian noise (standard deviation) | 5 | 10 | 20 | 35 | 50 |
| Contrast factor | 0.8 | 0.6 | 0.45 | 0.3 | 0.2 |
| Brightness up (grey levels) | 15 | 30 | 50 | 70 | 90 |
| Brightness down (grey levels) | 15 | 30 | 50 | 70 | 90 |
| JPEG quality | 60 | 40 | 25 | 15 | 8 |
| Downscale factor | 0.75 | 0.5 | 0.35 | 0.25 | 0.15 |

## Synthetic gallery

The required synthetic gallery is expected at:

`results/robustness/brahmi/gallery_synthetic.png`


### Generate the demo gallery

From the repository root, run:

```powershell
backend\.venv\Scripts\python.exe -m research.robustness.brahmi_corruptions.run --demo
```


**Important:** These are synthetic corruptions used for controlled experiments. They are not intended to reproduce every aspect of real scanner behaviour.

## Run tests

From the repository root, run:

```powershell
backend\.venv\Scripts\python.exe -m pytest research\robustness\brahmi_corruptions\tests\test_corruptions.py -v
```

## Disclaimer

For research and educational use only. This library is not a clinical or diagnostic tool.