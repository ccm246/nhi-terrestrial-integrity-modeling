"""Compare two landscape runs: departure maps side by side plus response R2 dumbbells.

Usage:
    python -m scripts.tools.compare_runs RUN_A RUN_B --output FIG.png

RUN_x is a run name such as ``central_india`` or ``central_india_boundary``. Panels:
departure-percentile maps for both runs' predictors-only models (same color scale),
held-out R2 per response for A vs B, and predictors-only vs all-variables R2 for B.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#d9d8d2"
RAMP = LinearSegmentedColormap.from_list("departure", ["#e8f1fb", "#9cc3ee", "#2a78d6", "#0b2f5e"])


def load_map(run: str):
    folder = ROOT / "outputs" / "reference_condition_inference" / f"{run}_predictors_only_2018"
    stem = f"{run}_predictors_only_2018"
    with rasterio.open(folder / f"{stem}_reference_departure_percentile.tif") as src:
        percentile = src.read(1, masked=True).astype("float32")
        bounds = src.bounds
    with rasterio.open(folder / f"{stem}_inference_status.tif") as src:
        inside_aoi = ~np.ma.getmaskarray(src.read(1, masked=True))
    return percentile, inside_aoi, bounds


def block_reduce(array, valid, block=4):
    """Mean of valid pixels per block x block cell (NaN where none)."""
    rows, cols = (array.shape[0] // block) * block, (array.shape[1] // block) * block
    values = np.where(valid, array.filled(0), 0.0)[:rows, :cols]
    weights = valid[:rows, :cols].astype("float32")
    sums = values.reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    counts = weights.reshape(rows // block, block, cols // block, block).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan), counts


def r2_table(run: str, suffix: str) -> pd.Series:
    path = ROOT / "outputs" / "integrity_parameters" / f"{run}_{suffix}" / "response_metrics.csv"
    frame = pd.read_csv(path)
    return frame.set_index("display_name")["overall_weighted_r2"]


def dumbbell(ax, first: pd.Series, second: pd.Series, labels, colors, order):
    names = [n for n in order if n in first.index and n in second.index]
    y = np.arange(len(names))[::-1]
    ax.hlines(y, [first[n] for n in names], [second[n] for n in names], color=GREY, lw=2, zorder=1)
    ax.scatter([first[n] for n in names], y, s=46, color=colors[0], label=labels[0], zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.scatter([second[n] for n in names], y, s=46, color=colors[1], label=labels[1], zorder=3,
               edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.axvline(0, color="#52514e", lw=0.8)
    ax.set_xlim(-0.5, 1.02)
    ax.set_xlabel("Held-out R² (spatial cross-validation)", fontsize=9)
    ax.grid(axis="x", color="#ecebe6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.02))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--labels", nargs=2, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [args.run_a, args.run_b]

    fig = plt.figure(figsize=(14, 10.5), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], width_ratios=[1, 1, 0.06],
                            hspace=0.28, wspace=0.12)
    maps = {}
    for column, (run, label) in enumerate(zip((args.run_a, args.run_b), labels)):
        percentile, aoi, bounds = load_map(run)
        maps[run] = (percentile, aoi, bounds, label)
    west = min(m[2].left for m in maps.values()); east = max(m[2].right for m in maps.values())
    south = min(m[2].bottom for m in maps.values()); north = max(m[2].top for m in maps.values())
    image = None
    for column, run in enumerate((args.run_a, args.run_b)):
        percentile, aoi, bounds, label = maps[run]
        valid = ~np.ma.getmaskarray(percentile)
        reduced, counts = block_reduce(percentile, valid)
        aoi_reduced, _ = block_reduce(np.ma.masked_array(np.ones(aoi.shape, "float32")), aoi)
        ax = fig.add_subplot(grid[0, column])
        extent = (bounds.left / 1000, bounds.right / 1000, bounds.bottom / 1000, bounds.top / 1000)
        ax.imshow(np.where(np.isnan(aoi_reduced), np.nan, 1.0), extent=extent, cmap="Greys",
                  vmin=0, vmax=6, interpolation="nearest")
        image = ax.imshow(reduced, extent=extent, cmap=RAMP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xlim(west / 1000, east / 1000); ax.set_ylim(south / 1000, north / 1000)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        area = int(valid.sum()) * 0.25
        ax.set_title(f"{label}\n{area:,.0f} km² of grassland scored; "
                     f"mean departure {np.nanmean(percentile.filled(np.nan)):.2f}",
                     fontsize=10.5, loc="left")
    cax = fig.add_subplot(grid[0, 2])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Departure from modeled reference (percentile; dark = farther)", fontsize=9)
    colorbar.outline.set_visible(False)

    order = list(r2_table(args.run_b, "predictors_only").sort_values(ascending=False).index)
    ax = fig.add_subplot(grid[1, 0])
    dumbbell(ax, r2_table(args.run_a, "predictors_only"), r2_table(args.run_b, "predictors_only"),
             labels, (BLUE, ORANGE), order)
    ax.set_title("Same predictors-only models, two extents", fontsize=10.5, loc="left")
    ax = fig.add_subplot(grid[1, 1])
    dumbbell(ax, r2_table(args.run_b, "predictors_only"), r2_table(args.run_b, "all_variables"),
             ("Environmental predictors only", "All variables"), (BLUE, ORANGE), order)
    ax.set_title(f"{labels[1]}: predictors only vs all variables", fontsize=10.5, loc="left")
    ax.set_yticklabels([])
    fig.suptitle("Central India grassland: reference-condition models", fontsize=14, x=0.07,
                 ha="left", y=0.98)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
