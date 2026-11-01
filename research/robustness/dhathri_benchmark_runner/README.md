# Benchmark Runner – Group B Robustness

## Overview
The benchmark runner evaluates image classification performance under different image-quality corruption conditions. It applies corruption functions to input images, runs predictions, calculates correctness, records quality scores, and saves the results in CSV format.

## Implemented Features
- Baseline image evaluation.
- Integration of seven corruption functions from the Brahmi corruption library.
- Five corruption severity levels (1–5).
- Prediction and correctness recording.
- Quality-score recording.
- CSV result generation.
- Incremental saving and basic resume functionality.

## Corruption Types
1. Blur
2. Noise
3. Contrast
4. Brightness up
5. Brightness down
6. JPEG compression
7. Downscale

## Demo Testing
The pipeline was tested using eight synthetic demo images.

- Total conditions per image: 36
- Total generated rows: 288
- Baseline rows: 8
- Rows per corruption type: 40

The demo used placeholder prediction and quality-score functions for pipeline testing only. These results are not research findings.

## Running the Benchmark
Run from the repository root:

```powershell
python -m research.robustness.dhathri_benchmark_runner.run
```

## Dependencies
- Python
- NumPy
- Pandas
- Pillow
- OpenCV
- Pytest

## Research Requirements
Actual research evaluation requires the approved frozen model checkpoint, real test images and labels, and the existing patient-level data split. The model and preprocessing must remain consistent throughout the experiment.
