# SDSS: Spectral Analysis - Project 1

Cserpák Mihály Márkó, Neptun ID: AD0FHH.

Predict SDSS DR17 galaxy redshift and the `g`, `r` and `i` cModel magnitudes
from optical spectra. The notebook compares linear regression and random forest
regression with the original prepared inputs and with two principal components.

- [Report (PDF)](project1.pdf)
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
- `figures/`: figures required by the LaTeX report and produced by the notebook.
- `scripts/`: supporting plot and quality-audit scripts.

The prepared data used by the models are included. The original FITS spectra
occupy approximately 5.4 GiB and are downloaded from SDSS by the notebook rather
than stored in Git. Trained forest caches, the virtual environment, machine
logs and temporary build files are also excluded. Parameter searches regenerate
their model caches when run again.


