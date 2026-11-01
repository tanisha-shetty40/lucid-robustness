\# Lucid Reliability Suite – Group B: Robustness Results



\## 1. Objective



The objective of this module is to investigate how image-quality degradation affects the performance of the Lucid pneumonia classification model.



The analysis considers different image corruption types and severity levels. It evaluates whether model performance decreases as image quality deteriorates and whether the image quality score can help predict this degradation.



\## 2. Experimental Setup



A synthetic demonstration dataset containing 200 images was generated for initial implementation and testing.



The dataset covers seven corruption types:



\- Blur

\- Gaussian noise

\- Contrast adjustment

\- Brightness increase

\- Brightness decrease

\- JPEG compression

\- Downscaling



Each corruption type includes six severity levels, including the unchanged baseline.



The analysis evaluates two performance metrics:



1\. \*\*Accuracy:\*\* Measures the proportion of correctly classified images.

2\. \*\*ROC-AUC:\*\* Measures the model's ability to distinguish between the two classes.



\## 3. Statistical Methodology



Bootstrap resampling is used to estimate confidence intervals for the performance metrics.



The implementation uses:



\- Number of bootstrap resamples: 500

\- Random seed: 42

\- Confidence interval: 95%

\- Resampling unit: Individual image rows



For each corruption type and severity level, the metric is calculated on the original group and on bootstrap samples. The 2.5th and 97.5th percentiles of the bootstrap estimates are used as the confidence interval bounds.



This provides an estimate of uncertainty around the measured performance.



\## 4. Generated Visualisations



The demo execution successfully generated the following outputs:



\### Accuracy vs Severity



`fig\_accuracy\_vs\_severity.png`



Shows the accuracy estimates and bootstrap confidence intervals across corruption types and severity levels.



\### ROC-AUC vs Severity



`fig\_auc\_vs\_severity.png`



Shows the ROC-AUC estimates and bootstrap confidence intervals across corruption types and severity levels.



\### Synthetic Image Gallery



`fig\_gallery\_synthetic.png`



Provides illustrative examples of image-quality degradation. These are synthetic examples, not actual chest X-ray images.



\## 5. Initial Implementation Status



The synthetic demo completed successfully and generated:



\- Accuracy summary CSV

\- ROC-AUC summary CSV

\- Accuracy versus severity plot

\- ROC-AUC versus severity plot

\- Synthetic image gallery



The initial bootstrap unit tests also passed, covering confidence interval behaviour, reproducibility with a fixed seed, and handling of single-class AUC groups.



\*\*Note:\*\* The current demo is intended to validate the analysis pipeline. Its synthetic results are not evidence of the actual pneumonia classifier's robustness.



\## 6. Quality Score Analysis



An important objective is to determine whether the image quality score predicts classification performance under degraded image conditions.



The supplied sweep format includes an overall `quality` score and individual quality-related measurements:



\- Brightness

\- Contrast

\- Sharpness

\- Noise

\- Clipping



The relationship between these quality measurements and model performance still needs to be evaluated using the actual sweep dataset.



\## 7. Limitations and Future Work



The current results are based on synthetic demonstration data. Therefore, actual model degradation and the predictive relationship between image quality and classification performance cannot yet be concluded.



The next steps are:



1\. Run the analysis on the real robustness sweep CSV.

2\. Compare accuracy and ROC-AUC across corruption types and severity levels.

3\. Examine whether lower quality scores are associated with increased classification errors.

4\. Evaluate how well the quality score predicts model performance.

5\. Interpret the results and document the findings using real experimental data.



\## Conclusion



The Group B robustness analysis pipeline has been implemented and successfully tested using synthetic demo data. It supports bootstrap confidence interval estimation, accuracy and ROC-AUC evaluation, and visualisation across multiple image corruption conditions.



The final robustness conclusions will be established after analysing the real sweep dataset.

