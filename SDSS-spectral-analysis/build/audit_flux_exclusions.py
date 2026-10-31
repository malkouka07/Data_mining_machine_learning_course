"""Count the existing pixel exclusion policy across every downloaded spectrum."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import ast
import json

from astropy.io import fits
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/dr17_5000"
configuration = json.loads((DATA / "prepared_baseline/configuration.json").read_text(encoding="utf-8"))
manifest = pd.read_csv(DATA / "spectra_manifest.csv", dtype={"specObjID": str})
catalog = pd.read_csv(DATA / "galaxies.csv", dtype={"specObjID": str})
assert len(manifest) == len(catalog) == 5000
assert manifest.specObjID.is_unique and catalog.specObjID.is_unique
assert set(manifest.specObjID) == set(catalog.specObjID)
assert manifest.status.eq("ok").all()

notebook = nbformat.read(ROOT / "project1.ipynb", as_version=4)
flag_definitions = None
for cell in notebook.cells:
    if cell.cell_type != "code" or "SPPIXMASK_NAMES =" not in cell.source:
        continue
    for node in ast.parse(cell.source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SPPIXMASK_NAMES" for target in node.targets
        ):
            flag_definitions = ast.literal_eval(node.value)
if flag_definitions is None:
    raise RuntimeError("Could not find the notebook's spectral flag definitions")
flag_bits = {name: bit for bit, name in flag_definitions.items()}
excluded_bits = sum(1 << flag_bits[name] for name in configuration["excluded_and_flags"])
start, end = configuration["shared_grid_start"], configuration["shared_grid_end"]
step = configuration["loglam_step"]


def audit(row):
    with fits.open(ROOT / row.path, memmap=False) as hdul:
        spectrum = hdul[1].data
        flux = spectrum["flux"].astype(float)
        ivar = spectrum["ivar"].astype(float)
        and_mask = spectrum["and_mask"].astype(np.int64)
        positions = np.rint(spectrum["loglam"].astype(float) / step).astype(np.int64)
        invalid = ~np.isfinite(flux) | ~np.isfinite(ivar) | (ivar <= 0)
        flagged = (and_mask & excluded_bits) != 0
        excluded = invalid | flagged
        shared = (positions >= start) & (positions <= end)
        if shared.sum() != end - start + 1:
            raise ValueError(f"Unexpected shared pixel count for {row.specObjID}")
        records = []
        for coverage, keep in [("Native coverage", np.ones(len(flux), dtype=bool)),
                               ("Shared range", shared)]:
            records.append({
                "specObjID": row.specObjID, "coverage": coverage,
                "pixels": int(keep.sum()),
                "numerically_invalid": int(np.count_nonzero(invalid & keep)),
                "selected_AND_flags": int(np.count_nonzero(flagged & keep)),
                "both_reasons": int(np.count_nonzero(invalid & flagged & keep)),
                "excluded_pixels": int(np.count_nonzero(excluded & keep)),
            })
    return records


rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(audit, row) for row in manifest.itertuples()]
    for count, future in enumerate(as_completed(futures), start=1):
        rows.extend(future.result())
        if count % 500 == 0:
            print(f"Checked {count:,}/5,000 spectra", flush=True)
detail = pd.DataFrame(rows).sort_values(["coverage", "specObjID"])
summary = detail.groupby("coverage", sort=False)[[
    "pixels", "numerically_invalid", "selected_AND_flags", "both_reasons", "excluded_pixels"
]].sum()
summary["excluded_percent"] = 100 * summary.excluded_pixels / summary.pixels
summary["galaxies_checked"] = detail.groupby("coverage").specObjID.nunique()
summary["galaxies_with_excluded_pixels"] = detail.loc[detail.excluded_pixels > 0].groupby("coverage").specObjID.nunique()
assert summary.loc["Shared range", "pixels"] == 5000 * (end - start + 1)
np.testing.assert_array_equal(summary.excluded_pixels,
                              summary.numerically_invalid + summary.selected_AND_flags - summary.both_reasons)
OUTPUT = DATA / "quality_audit"
OUTPUT.mkdir(exist_ok=True)
detail.to_csv(OUTPUT / "flux_exclusions_by_galaxy.csv", index=False)
summary.to_csv(OUTPUT / "flux_exclusion_summary.csv")
(OUTPUT / "configuration.json").write_text(json.dumps({
    "spectra_checked": 5000, "excluded_AND_flags": configuration["excluded_and_flags"],
    "numerical_exclusions": "nonfinite flux or inverse variance, or inverse variance <= 0",
    "excluded_bits": excluded_bits, "shared_grid_start": start, "shared_grid_end": end,
    "loglam_step": step, "counts_before_galaxy_and_column_rejection": True,
}, indent=2), encoding="utf-8")
print(summary.to_string(), flush=True)
