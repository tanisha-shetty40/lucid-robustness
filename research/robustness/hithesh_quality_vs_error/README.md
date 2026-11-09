# Quality vs. Error Analysis Core Module

This research module evaluates how well image quality metrics correlate with corruption severity, model prediction errors, and model accuracy under sample rejection.

---

## 1. Core Research Questions & Findings

### RQ1: Does the quality score drop as corruption severity increases?
* **Finding:** Yes. As image corruption levels increase, the estimated quality metric degrades monotonically. 
* **Details:** Quality vs. Severity boxplots (`fig_quality_vs_severity.png`) demonstrate a clear decline in quality distributions as severity scales from level 1 through level 5.

### RQ2: Does the quality score accurately predict wrong predictions per image?
* **Finding:** Yes, quality scores serve as a strong predictive signal for model errors.
* **Details:** Evaluated using Area Under the ROC Curve (AUC) for error prediction:
  * **AUC:** `0.9273` (95% CI: `[0.9021, 0.9472]`)
  * **Interpretation:** High AUC indicates that lower quality scores are consistently associated with incorrect predictions.

### RQ3: Does rejecting low-quality images actually improve overall accuracy?
* **Finding:** Yes, selective classification based on quality thresholds improves model accuracy on the remaining sample set.
* **Details:** As lower-quality images are rejected (moving from 100% coverage down to 50% coverage in `tables.csv` and `fig_rejection.png`), accuracy increases continuously, demonstrating that quality metrics can effectively filter out error-prone samples.

---

## 2. Usage & Reproduction

To regenerate all analysis outputs, metrics, and plots in a single command, run:

```bash
# Run using generated demo sweep data
python -m research.robustness.hithesh_quality_vs_error.run --demo

# Run using a full sweep dataset
python -m research.robustness.hithesh_quality_vs_error.run --sweep path/to/sweep.csv