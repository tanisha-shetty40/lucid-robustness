# Group B - Model Robustness Research



## Purpose

Evaluate how image-quality degradation affects the performance of the frozen image-classification model and investigate whether image-quality scores are associated with model errors.



## Research Plan

See \[PLAN.md](PLAN.md) for the objectives, corruption conditions, evaluation protocol, metrics, and completion criteria.



## Planned Work

- Verify the approved model checkpoint and test-split configuration.

- Evaluate baseline performance on the existing test split.

- Apply controlled image-quality corruptions.

- Measure classification performance under each condition.

- Analyze the relationship between image-quality scores and prediction errors.

- Document results, limitations, and reproducibility details.



## Research Integrity

- Preserve the approved frozen model and existing data splits.

- Do not use DEMO predictions or synthetic smoke-test metrics as research results.

- Record corruption parameters and evaluation configuration.

- Do not report experimental results until the approved checkpoint and evaluation setup are verified.



## Scope

This is a research and educational project, not a diagnostic system.




## Completed Work — Real Robustness Benchmark

The real-data robustness benchmark has been completed using the verified frozen ResNet18 checkpoint and the original held-out test split.

- Test images evaluated: 874.
- Corruption types: 7.
- Severity levels: 5 per corruption.
- Total evaluation conditions: 36, including baseline.
- Prediction records: 31,464.
- Bootstrap resamples: 500 per metric and condition.
- Metrics: Accuracy, sensitivity, specificity, and ROC-AUC.
- Confidence intervals: 95% bootstrap intervals.

### Baseline Results

| Metric | Result |
|---|---:|
| Accuracy | 98.17% |
| Sensitivity | 98.15% |
| Specificity | 98.21% |
| ROC-AUC | 99.79% |

### Results and Reproducibility

- Full report: `results/robustness/real_benchmark/REAL_BENCHMARK_REPORT.md`
- Raw predictions: `results/robustness/real_benchmark/sweep.csv`
- Metric summaries and plots: `results/robustness/real_benchmark/analysis/`
- Benchmark runner: `research/robustness/real_benchmark/real_sweep.py`

The benchmark evaluates controlled image corruptions and is intended for research and educational use only. It is not a clinical diagnostic system.

**Remaining research work:** Analyze the relationship between image-quality scores and model prediction errors, as specified in the original research plan.
