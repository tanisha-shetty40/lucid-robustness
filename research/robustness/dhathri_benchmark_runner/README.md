# Dhathri Benchmark Runner

## Purpose
This module tests how image-quality degradation affects the performance of the medical image classification model.

## Main Tasks
- Apply image corruptions at different severity levels.
- Obtain model predictions for each image.
- Calculate image-quality scores.
- Save benchmark results in CSV format.
- Summarize model performance using accuracy, sensitivity, specificity and AUC.

## Temporary Testing
Until the actual corruption library is available, identity and brightness-shift functions will be used for testing.

## Files
- `sweep.py` – Runs image corruption and prediction tests.
- `summarize.py` – Calculates performance metrics.
- `run.py` – Starts the benchmark.
- `make_demo_data.py` – Generates sample testing data.
- `tests/test_sweep.py` – Tests the sweep functionality.
