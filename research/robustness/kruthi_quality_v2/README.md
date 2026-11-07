\# Kruthi: Quality Score v2



\## Problem

This module explores whether five image-quality indicators can help estimate when a model prediction may be wrong.



\## Current status

The indicator, scoring, fitting, synthetic-data, and demo-runner files are being prepared. No research results have been produced.



\## Indicators

\- Histogram spread: the range between the 1st and 99th brightness percentiles.

\- Edge density: the fraction of pixels marked as edges.

\- Entropy: the variety of gray levels in the image.

\- Blockiness: how much pixel jumps differ at 8-pixel block boundaries.

\- Local contrast: average brightness variation within 16 x 16 pixel tiles.



\## Demo

The demo uses synthetic geometric shapes only. Its output is not a medical-image result and must not be reported as a research finding.



```powershell

$env:PYTHONPATH="backend;."

python -m research.robustness.kruthi\_quality\_v2.run --demo

