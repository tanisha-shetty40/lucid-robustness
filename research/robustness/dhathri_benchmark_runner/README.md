# Benchmark Runner - Group B Robustness

## Overview

This project evaluates the robustness of a frozen ResNet-18 chest X-ray classification model under different image-quality corruption conditions.

The full classifier benchmark was completed using 874 official test images from the approved dataset split.

This is a research and educational robustness evaluation, not a medical diagnostic tool.

## Implemented Features

- Integration of the approved frozen ResNet-18 classifier.
- Evaluation using 874 official test images.
- Seven corruption types from the Brahmi corruption library.
- Five corruption severity levels (1-5).
- Baseline evaluation (severity 0).
- Model predictions and pneumonia probabilities.
- Classification correctness calculation.
- Image-quality measurements.
- CSV result generation.
- Deterministic random seeds.
- Resume support.
- Command-line interface (CLI).
- Demo mode and stub mode.
- Automated tests: 5 passed.
- Summary generation using accuracy, sensitivity, specificity and AUC.

## Corruption Types

1. Blur
2. Noise
3. Contrast
4. Brightness increase
5. Brightness decrease
6. JPEG compression
7. Downscaling

## Full Classifier Benchmark - 874 Images

- Dataset split: Official test split
- Images: 874
- Model: ResNet-18
- Checkpoint: `D:\Projects\Lucid\models\classifier.pt`
- Classes: NORMAL and PNEUMONIA
- Corruptions: 7
- Severity levels: 1-5
- Baseline severity: 0
- Conditions per image: 36
- Total rows: 31,464
- Prediction mode: REAL MODEL
- Correct predictions across all conditions: 30,001
- Pooled accuracy across all conditions: 95.35%

Note: The 95.35% value is pooled across baseline and corrupted conditions. It is not the clean-baseline accuracy.

### Preprocessing

- Convert grayscale images to three channels.
- Resize to 224 x 224.
- Convert to tensors.
- Apply ImageNet normalization.
- No CLAHE.

## Full Benchmark Results

Selected accuracy results:

| Blur severity | Accuracy |
|---|---:|
| 1 | 98.1% |
| 2 | 96.7% |
| 3 | 86.3% |
| 4 | 76.3% |
| 5 | 74.4% |

Blur showed a substantial accuracy reduction at higher severity levels. Contrast also reduced accuracy, reaching approximately 80.3% at severity 5.

Brightness changes, JPEG compression, noise and downscaling showed comparatively smaller accuracy changes in this experiment.

These findings describe model behaviour on this test dataset only.

## Generated Full Benchmark Files

- `sweep_full_classifier.csv` - Original full classifier results.
- `sweep.csv` - Rule Book formatted results.
- `sweep_none.csv` - Baseline results.
- `sweep_blur.csv`
- `sweep_noise.csv`
- `sweep_contrast.csv`
- `sweep_brightness_up.csv`
- `sweep_brightness_down.csv`
- `sweep_jpeg.csv`
- `sweep_downscale.csv`
- `summary.csv` - Accuracy, sensitivity, specificity and AUC by corruption and severity.
- `results.json` - Benchmark metadata.

The Rule Book formatted CSV files contain these 11 columns:

`image_id, label, corruption, severity, prob_pneumonia, quality, q_brightness, q_contrast, q_sharpness, q_noise, q_clipping`

The summary file contains:

`corruption, severity, n, auc, accuracy, sensitivity, specificity`

## Command-Line Interface (CLI)

The benchmark runner supports three modes:

1. **Real mode:** Uses the official test images, frozen classifier and approved image-quality function.
2. **Demo mode:** Uses eight synthetic images, placeholder predictions and demo quality measurements.
3. **Stub mode:** Uses eight synthetic images, placeholder predictions and two simple placeholder corruptions.

Run commands from the `Lucid-Team` repository root using the backend Python environment.

### Real mode

Run the benchmark using the real classifier:

```powershell
& "D:\Projects\Lucid\backend\.venv312\Scripts\python.exe" -m research.robustness.dhathri_benchmark_runner.run
```

Run with selected corruptions and a limited number of images:

```powershell
& "D:\Projects\Lucid\backend\.venv312\Scripts\python.exe" -m research.robustness.dhathri_benchmark_runner.run --corruptions blur noise --limit 50 --seed 42
```

### Demo mode

```powershell
& "D:\Projects\Lucid\backend\.venv312\Scripts\python.exe" -m research.robustness.dhathri_benchmark_runner.run --demo
```

- Images: 8
- Corruptions: 7
- Total rows: 288

### Stub mode

```powershell
& "D:\Projects\Lucid\backend\.venv312\Scripts\python.exe" -m research.robustness.dhathri_benchmark_runner.run --stub
```

- Images: 8
- Corruptions: identity and brightness_shift
- Total rows: 88

### CLI Arguments

| Argument | Description |
|---|---|
| `--corruptions` | Select corruption names |
| `--limit` | Maximum number of images |
| `--seed` | Set random seed (default: 42) |
| `--demo` | Run synthetic-image demo mode |
| `--stub` | Run placeholder stub mode |

The real model is used when neither `--demo` nor `--stub` is specified.

### CLI Output Files

CLI outputs are stored separately in:

`research/robustness/dhathri_benchmark_runner/cli_results/`

Main output files:

- `cli_demo_results.csv`
- `cli_stub_results.csv`
- `cli_real_results.csv`
- `sweep.csv`
- `sweep_none.csv`
- Individual corruption CSV files.

The CLI outputs are kept separate from the completed full benchmark results.

Note: The runner supports resuming previously completed image-condition combinations. Use a fresh output file or folder when starting a different experiment configuration so that results from different configurations are not mixed.

## CLI Verification

The following tests were completed successfully:

| Test | Images | Conditions | Rows |
|---|---:|---:|---:|
| Demo mode | 8 | 36 | 288 |
| Stub mode | 8 | 11 | 88 |
| Real-model smoke test (blur) | 2 | 6 | 12 |

The real-model smoke test confirmed:

- Official test images were loaded.
- The frozen classifier generated predictions.
- The approved image-quality function returned measurements.
- All 11 required CSV columns were present.
- No missing quality values were found.

The two-image real-model test is a CLI verification, not a replacement for the completed 874-image benchmark.

## Earlier 50-Image Validation

The earlier 50-image experiment is preserved separately:

- `sweep_50_classifier.csv`
- `classification_summary_50.csv`
- `classifier_metadata_50.json`
- `classifier_graphs/`

These preliminary results are separate from the completed 874-image benchmark.

## Provisional Quality Benchmark

The earlier quality-only experiment is preserved separately:

- `sweep_quality_provisional.csv`
- `quality_summary.csv`
- `results_metadata.json`
- Six graphs in `quality_graphs/`

Noise can increase a sharpness measurement because of intensity variations. A higher sharpness value does not necessarily mean a clearer image.

## Automated Tests

Five automated tests passed:

1. Sweep execution and expected row count.
2. Quality calculation on corrupted images.
3. Resume functionality.
4. Reproducibility with the same seed.
5. Summary accuracy and AUC calculation.

Test command:

```powershell
& "D:\Projects\Lucid\backend\.venv312\Scripts\python.exe" -m pytest research/robustness/dhathri_benchmark_runner/tests -v
```

Latest result: **5 passed in 1.25 seconds.**

## Research Requirements

- Use the approved frozen model checkpoint.
- Use official test images and labels.
- Preserve the existing patient-level data split.
- Keep model architecture and preprocessing consistent.
- Do not interpret classifier outputs as medical diagnoses.

## Limitations

- Results depend on the selected dataset, model and corruption functions.
- Demo outputs use synthetic data and placeholder predictions.
- Stub outputs use placeholder corruption functions.
- The earlier 50-image and provisional quality experiments are separate from the full benchmark.
- Exact full-run elapsed time was not recorded.

This project is intended for research and educational robustness evaluation only.