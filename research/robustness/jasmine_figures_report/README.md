\# Lucid Reliability Suite – Group B: Robustness



\## Overview



This module evaluates how image quality degradation affects the performance of the Lucid pneumonia classification model.



It analyses model accuracy and ROC-AUC across different image corruption types and severity levels using bootstrap confidence intervals.



\## Implemented Features



\- Bootstrap confidence interval calculation (500 resamples).

\- Fixed random seed (42) for reproducibility.

\- Accuracy and ROC-AUC evaluation.

\- Synthetic demo dataset generation using 200 images.

\- Performance plots across corruption types and severity levels.

\- Synthetic image quality gallery.

\- Command-line support for demo data and supplied sweep CSV files.



\## Supported Corruption Types



1\. Blur

2\. Gaussian noise

3\. Contrast adjustment

4\. Brightness increase

5\. Brightness decrease

6\. JPEG compression

7\. Downscaling



\## Project Structure



```text

jasmine\_figures\_report/

├── bootstrap.py

├── figures.py

├── make\_demo\_data.py

├── run.py

├── README.md

├── results\_page.md

├── tests/

│   └── test\_bootstrap.py

├── fig\_accuracy\_vs\_severity.png

├── fig\_auc\_vs\_severity.png

├── fig\_gallery\_synthetic.png

├── accuracy\_summary.csv

└── auc\_summary.csv

```



\## Setup



Run commands from the Lucid project root.



Set the Python module path:



```powershell

$env:PYTHONPATH="backend;."

```



Use the project virtual environment:



```powershell

.\\backend\\.venv\\Scripts\\python.exe

```



\## Running the Synthetic Demo



```powershell

$env:PYTHONPATH="backend;."

.\\backend\\.venv\\Scripts\\python.exe -m research.robustness.jasmine\_figures\_report.run --demo

```



The demo generates synthetic data for 200 images across seven corruption types and six severity levels.



\## Running with Supplied Sweep Data



```powershell

$env:PYTHONPATH="backend;."

.\\backend\\.venv\\Scripts\\python.exe -m research.robustness.jasmine\_figures\_report.run --sweep path\\to\\sweep.csv

```



The supplied CSV must contain:



\- `image\_id`

\- `label`

\- `corruption`

\- `severity`

\- `prob\_pneumonia`



Quality-related columns include `quality`, `q\_brightness`, `q\_contrast`, `q\_sharpness`, `q\_noise`, and `q\_clipping`.



\## Generated Outputs



| File | Description |

|---|---|

| `fig\_accuracy\_vs\_severity.png` | Accuracy across corruption severity |

| `fig\_auc\_vs\_severity.png` | ROC-AUC across corruption severity |

| `fig\_gallery\_synthetic.png` | Illustrative synthetic image examples |

| `accuracy\_summary.csv` | Accuracy estimates and confidence intervals |

| `auc\_summary.csv` | ROC-AUC estimates and confidence intervals |



\## Testing



Run the tests using:



```powershell

$env:PYTHONPATH="backend;."

.\\backend\\.venv\\Scripts\\python.exe -m pytest research/robustness/jasmine\_figures\_report -v

```



\## Important Note



The `--demo` mode uses synthetic data for development and testing. Its results are illustrative and must not be interpreted as actual pneumonia model robustness results.



Real robustness conclusions require the supplied image sweep data.

