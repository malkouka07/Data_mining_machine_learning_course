"""Plot five native downloaded spectra without smoothing, filtering or imputation."""

from pathlib import Path

from astropy.io import fits
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
manifest = pd.read_csv(ROOT / "data/dr17_5000/spectra_manifest.csv", dtype={"specObjID": str})
selection = manifest.loc[manifest.status == "ok"].head(5)
if len(selection) != 5:
    raise ValueError("Five downloaded spectra are required.")

with plt.rc_context({"font.size": 11, "axes.titlesize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "axes.spines.top": False,
                     "axes.spines.right": False}):
    fig, axes = plt.subplots(5, 1, figsize=(8.8, 8.8), sharex=True,
                             constrained_layout=True)
    for axis, row in zip(axes, selection.itertuples()):
        with fits.open(ROOT / row.path, memmap=False) as hdul:
            spectrum = hdul[1].data
            wavelength = 10.0 ** spectrum["loglam"].astype(float)
            flux = spectrum["flux"].astype(float)
        assert len(flux) == row.pixels
        assert np.isfinite(wavelength).all() and np.all(np.diff(wavelength) > 0)
        # Nonfinite fluxes remain gaps; finite negative values are retained.
        axis.plot(wavelength, flux, color="#275d85", linewidth=0.65)
        axis.set_title(f"Galaxy {row.specObjID}", loc="left")
        axis.grid(alpha=0.2)
        axis.margins(x=0.01, y=0.1)
    fig.supxlabel(r"Observed wavelength [$\mathrm{\AA}$]")
    fig.supylabel(r"Flux [$10^{-17}\,\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}$]")
    fig.savefig(ROOT / "figures/example_galaxy_spectra.png", dpi=250)
    plt.close(fig)
print("Saved native spectra plot for the first five downloaded galaxies.")
