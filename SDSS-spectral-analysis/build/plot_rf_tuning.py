"""Plot the saved validation trials without rerunning any model training."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/dr17_5000/random_forest_tuning/4b3527bb672c710d"
TARGETS = ["z", "cModelMag_g", "cModelMag_r", "cModelMag_i"]
STAGES = [
    ("stage1_tree_counts.csv", "n_estimators", "trees", 1),
    ("stage2_feature_sampling.csv", "max_features", "features", 100),
    ("stage3_leaf_sizes.csv", "min_samples_leaf", "leaves", 1),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
})

count = 0
for filename, parameter, suffix, scale in STAGES:
    trials = pd.read_csv(RESULTS / filename)
    for target in TARGETS:
        rows = trials.loc[trials.target == target].sort_values(parameter)
        expected = 4 if parameter == "min_samples_leaf" else 3
        if len(rows) != expected or rows[parameter].duplicated().any():
            raise ValueError(f"Unexpected trial count for {target}: {filename}")
        count += len(rows)
        x = rows[parameter].to_numpy() * scale
        y = rows.validation_MAE.to_numpy()
        fig, ax = plt.subplots(figsize=(2.18, 1.58), layout="constrained")
        ax.plot(x, y, color="#275d85", linewidth=1.25,
                marker="o", markersize=4, markerfacecolor="white")
        best = y.argmin()
        ax.plot(x[best], y[best], "o", color="#275d85", markersize=5)
        ax.set_xticks(x, [f"{value:g}" for value in x])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.tick_params(length=3, width=0.6)
        ax.grid(axis="y", color="0.88", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.13, y=0.24)
        fig.savefig(ROOT / "build" / f"rf_mae_{target}_{suffix}.pdf")
        plt.close(fig)

if count != 40:
    raise ValueError(f"Expected 40 stage trials, found {count}")
print("Saved 12 plots covering all 40 validation trials.")
