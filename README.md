# Benchmark Runner – Group B Robustness

## 1. Overview

This project evaluates the quality of chest X-ray images under different image-corruption conditions. The benchmark applies corruption functions at different severity levels and records image-quality measurements in CSV files.

The current implementation includes a 50-image provisional quality benchmark. Model prediction evaluation will be integrated after the approved frozen model checkpoint is provided.

## 2. Implemented Features

- Baseline image-quality evaluation.
- Integration of seven corruption functions from the Brahmi corruption library.
- Five corruption severity levels (1–5).
- Quality-score recording.
- Recording of brightness, contrast, sharpness, noise and clipping measurements.
- CSV result generation.
- Deterministic random seeds for reproducibility.
- Automated tests for sweep execution, CSV saving and resume functionality.
- Quality-summary generation and graph plotting.

## 3. Corruption Types

1. Blur
2. Noise
3. Contrast
4. Brightness increase
5. Brightness decrease
6. JPEG compression
7. Downscaling

## 4. Provisional Benchmark Results

The quality benchmark was tested using 50 real test images from the existing patient-level test split.

- Number of images: 50
- Corruption types: 7
- Severity levels: 1–5
- Baseline condition: Severity 0
- Total conditions per image: 36
- Total generated rows: 1,800

The benchmark generated:

- `sweep_quality_provisional.csv`
- `quality_summary.csv`
- Six quality graphs in the `quality_graphs` folder.
- `results_metadata.json`

These results describe image-quality changes only. They are provisional and should not be treated as model-performance findings.

**Note:** Noise can increase the sharpness measurement because the metric uses image intensity variations. A higher sharpness value does not necessarily mean that an image is clearer.

## 5. Running the Provisional Quality Benchmark

The script is:

`run_real_quality_sweep.py`

Run it using the configured project Python environment:

```powershell
& "D:\Projects\Lucid\backend\.venv312\Scripts\python.exe" "D:\Projects\Lucid-Team\research\robustness\dhathri_benchmark_runner\run_real_quality_sweep.py"
```

The script uses the existing test split, approved image-quality function and corruption library.

## 6. Running Automated Tests

Run the tests from the repository root:

```powershell
python -m pytest research/robustness/dhathri_benchmark_runner/tests -v
```

The current test suite contains three tests covering sweep execution, CSV saving and resume functionality.

## 7. Pending Work

- Obtain the approved frozen model checkpoint.
- Integrate the actual model prediction function.
- Record pneumonia probabilities and classification results.
- Validate the model-integrated benchmark using 50 images.
- Generate classification-performance summaries.
- Complete the full test-set evaluation after validating the 50-image run.

## 8. Research Requirements

The final research evaluation requires the approved frozen model checkpoint, real test images and labels, and the existing patient-level data split.

The same model and preprocessing must be used throughout the experiment. Synthetic demo results and provisional quality-only results must not be reported as final model-performance findings.
