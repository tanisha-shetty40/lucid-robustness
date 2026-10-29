\# Group B — Robustness Research Plan



\*\*Project:\*\* Lucid — Medical Imaging Analysis \& Quality Platform  

\*\*Group:\*\* B — Robustness  

\*\*Status:\*\* Planning  

\*\*Purpose:\*\* Research and educational use only; not a diagnostic system.



\## 1. Objective



Evaluate how image-quality degradation affects the performance and reliability of the frozen image-classification model. Investigate whether image-quality scores are associated with model errors under different image-corruption conditions.



\## 2. Research Questions



1\. How does classification performance change when test images are degraded?

2\. Which corruption types and severity levels cause the largest performance degradation?

3\. Is the image-quality score associated with the probability of a classification error?

4\. Can image-quality measurements help identify conditions under which model predictions become less reliable?



\## 3. Experimental Scope



Evaluate the following corruption types:



\- Gaussian blur

\- Gaussian noise

\- Contrast reduction

\- Brightness increase

\- Brightness decrease

\- JPEG compression

\- Downscaling followed by upscaling



Use the original, uncorrupted test images as the baseline. Define and document the severity levels for each corruption before running the final experiment.



The planned sweep is 874 test images across 36 conditions, for a target of 31,464 image-condition records. Confirm the final condition count and test-set size against the team's approved experiment configuration before execution.



\## 4. Experimental Protocol



1\. Obtain the approved, frozen model checkpoint from the team's shared storage.

2\. Verify the checkpoint version and record its checksum if available.

3\. Use the existing patient-level test split at `data/processed/splits.json`. Do not regenerate or modify the split.

4\. Confirm the image preprocessing, input dimensions, data types, and pixel-value ranges expected by the model.

5\. Run inference on the original test images to establish baseline performance.

6\. Apply each defined corruption at its specified severity while preserving the original image labels and sample identifiers.

7\. Run inference on the corrupted images using the same frozen checkpoint and consistent preprocessing.

8\. Calculate performance metrics for each corruption condition.

9\. Compare the results against the original-image baseline.

10\. Analyse the relationship between image-quality scores and classification errors.

11\. Record failures, excluded samples, and deviations from the protocol.



\## 5. Evaluation Metrics



Report the following metrics where supported by the dataset and task:



\- Accuracy

\- Precision

\- Recall

\- F1-score

\- Confusion matrix

\- Number and proportion of classification errors

\- Change in each metric relative to the original-image baseline



For the image-quality analysis, record the quality score, corruption type, severity, sample identifier, ground-truth label, predicted label, and whether the prediction was correct.



Document the metric definitions and averaging method. Include class-wise results where appropriate, particularly if the dataset is imbalanced.



\## 6. Quality Score and Error Analysis



Investigate whether lower image-quality scores are associated with a higher rate of classification errors.



\- Compare quality-score distributions for correct and incorrect predictions.

\- Examine error rates across quality-score ranges.

\- Calculate an appropriate association measure.

\- Report uncertainty or confidence intervals where feasible.

\- Distinguish association from causation.



Do not claim that quality scores predict errors reliably without supporting quantitative evidence.



\## 7. Reproducibility and Data Integrity



\- Use the approved frozen checkpoint; do not retrain or replace it for the primary experiment.

\- Preserve the existing test split and sample identifiers.

\- Record the software environment, random seeds where applicable, corruption parameters, and evaluation configuration.

\- Keep the original test images unchanged.

\- Do not commit medical images, datasets, or model checkpoints to Git.

\- Store only the minimum identifiers and results needed for analysis, following the project's privacy requirements.

\- Keep experiment outputs separate from application code.

\- Clearly document missing files, failed runs, and any deviations from the protocol.



\## 8. Validity Requirements



The primary experiment must use real test-set samples and actual predictions from the approved checkpoint.



The application’s DEMO mode must not be used to produce research predictions. Synthetic sample data generated for smoke testing must not be used to report model performance or robustness results.



If the approved checkpoint is unavailable, limit work to planning, code review, corruption visualization, and pipeline validation. Do not report final model-performance findings until the checkpoint is available.



\## 9. Planned Deliverables



1\. `research/robustness/PLAN.md` — approved experiment protocol.

2\. `research/robustness/README.md` — instructions for reproducing the experiments.

3\. Corruption gallery — representative examples of each corruption and severity.

4\. Experiment configuration — documented corruption parameters and evaluation settings.

5\. Results table — baseline and per-condition metrics.

6\. Error analysis — relationship between quality scores and prediction errors.

7\. Final research summary — findings, limitations, and recommendations.



\## 10. Completion Criteria



The study is complete when:



\- The experiment configuration and severity levels are documented.

\- The frozen checkpoint and existing test split are verified.

\- All planned conditions have been run or exceptions are documented.

\- Baseline and corrupted-image metrics have been calculated consistently.

\- Quality-score and error analyses have been completed.

\- Results are reproducible and limitations are clearly stated.

\- No DEMO predictions or synthetic smoke-test metrics are presented as research findings.



\## 11. Current Blocker



The local repository contains `data/processed/splits.json`, but `models/classifier.pt` is currently absent. Obtain and verify the approved checkpoint from the team's shared storage before running the primary robustness evaluation.



\*\*Safety note:\*\* Lucid is a research and educational platform, not a diagnostic system. Robustness findings do not establish clinical suitability.

