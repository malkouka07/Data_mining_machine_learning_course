# SDSS: Spectral Analysis - Project 1

Cserpák Mihály Márkó, Neptun ID: AD0FHH.

Predict SDSS DR17 galaxy redshift and the `g`, `r` and `i` cModel magnitudes
from optical spectra. The notebook compares linear regression and random forest
regression with the original prepared inputs and with two principal components.

- [Report (PDF)](build/project1.pdf)
- [Report source (LaTeX)](project1.tex)
- [Executed notebook](project1.ipynb)
- [SDSS DR17 data source](https://www.sdss4.org/dr17/)

## Files and data

- `data/dr17_5000/galaxies.csv`, `targets.csv`, `selection.sql`: the selected
  5,000 galaxies, four targets and the original catalog query.
- `data/dr17_5000/spectra_manifest.csv`: download paths and validation status.
- `data/dr17_5000/prepared_baseline/`: prepared training, validation and test
  arrays, galaxy IDs, the fitted median imputer and preprocessing records.
  The retained sample has 4,799 galaxies: 3,359 training, 720 validation and
  720 test galaxies. Inputs contain 1,902 flux columns and 815 missing indicators.
- `data/dr17_5000/pca/`: the fitted PCA transformation, component scores,
  explained-variance tables, validation results, parameter searches and timings.
- `data/dr17_5000/random_forest_tuning/`: the original sequential search results
  and validation predictions.
- `data/dr17_5000/quality_audit/`: measurement-exclusion counts across all
  5,000 downloaded spectra.
- `rf_tuning.py`, `regression_benchmarks.py`: search and timing helpers.
- `run_rf_tuning.py`, `run_pca_experiments.py`: runners for background execution.
- `build/`: the report, figures required by LaTeX, and plot/audit scripts.

The prepared data used by the models are included. The original FITS spectra
occupy approximately 5.4 GiB and are downloaded from SDSS by the notebook rather
than stored in Git. Trained forest caches, the virtual environment, machine
logs and temporary build files are also excluded. Parameter searches regenerate
their model caches when run again.

## Run the project

Run commands from this project folder. The original environment used Python
3.13 and scikit-learn 1.9.1. On Windows/PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name sdss-spectral-analysis --display-name "SDSS (.venv)"
```

Open `project1.ipynb` and select the `SDSS (.venv)` kernel. To reproduce the full
data inspection and preparation, run from the beginning. The spectrum download
resumes from already downloaded and validated files. Later PCA and efficiency
sections load the included prepared arrays directly; they do not require the
FITS files. The test split is saved but has not been evaluated in the report.

The prepared arrays can also be loaded independently:

```python
import numpy as np

with np.load("data/dr17_5000/prepared_baseline/regression_inputs.npz",
             allow_pickle=False) as data:
    X_train = data["X_train"]
    y_train = data["y_train"]
    feature_names = data["feature_names"]
    target_names = data["target_names"]
```

The file also includes validation/test arrays and galaxy IDs for all splits.
The matching feature and target row orders must be retained.

## Build the report

With `pdflatex` installed, run from this folder:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build project1.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build project1.tex
```

The report contains an AI usage statement. A matching statement is included at
the end of the notebook.
