# Group B - Model Robustness Research

## Purpose

Evaluate how image-quality degradation affects the performance of the frozen image-classification model and investigate whether image-quality scores are associated with model errors.

## Planned Work

- Verify the approved model checkpoint and test-split configuration.
- Evaluate baseline performance on the existing test split.
- Apply controlled image-quality corruptions.
- Measure classification performance under each condition.
- Analyze the relationship between image-quality scores and prediction errors.
- Document results, limitations, and reproducibility details.

## Completed Work

### Quality vs. Error Robustness Analysis (`hithesh_quality_vs_error`)

- **Contributor:** Hithesh K (`b/hithesh/quality-vs-error`)
- **Key Findings:**
  - **RQ1 (Degradation Across Severities):** Quantified model performance drop across progressive perturbation severity levels.
  - **RQ2 (Error Detection via Quality Scores):** Demonstrated strong predictive separation between correct and erroneous predictions using image-quality metrics (**ROC-AUC = 0.9273**).
  - **RQ3 (Selective Rejection Trade-offs):** Evaluated accuracy-coverage curves and rejection thresholds to filter low-quality inputs.
- **Code & Data:** [`research/robustness/hithesh_quality_vs_error/`](research/robustness/hithesh_quality_vs_error/)
  - `run.py` — Reproducible evaluation, corruption sweep, and metric generation pipeline.
  - `sweep.csv` — Full per-sample corruption sweep dataset.
  - `README.md` — Detailed methodology, edge cases, and reproduction commands.
- **Results & Figures:** [`results/robustness/hithesh/`](results/robustness/hithesh/)
  - `tables.csv` & `quality_error_auc.json` — Summary metrics and AUC scores.
  - `summary.md` — Written analysis of findings.
  - `fig_quality_vs_severity.png`, `fig_error_auc.png`, & `fig_rejection.png` — Visualizations of severity degradation, ROC-AUC performance, and selective rejection curves.

## Research Integrity

- Preserve the approved frozen model and existing data splits.
- Do not use DEMO predictions or synthetic smoke-test metrics as research results.
- Record corruption parameters and evaluation configuration.
- Do not report experimental results until the approved checkpoint and evaluation setup are verified.

## Scope

This is a research and educational project, not a diagnostic system.
