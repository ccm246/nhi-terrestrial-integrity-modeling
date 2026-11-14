"""Integrity maps restricted to pixels the reference can speak for.

Usage:
    python -m scripts.tools.make_assessable_maps OUTPUT_DIR SIMILARITY_DIR RUN [RUN ...]

A pixel is assessable when its climate, soil and terrain lie inside the reference's own range
(abiotic similarity > 0.05, i.e. not past the 95th-percentile reference distance; this reproduces
the "outside reference range" share of make_similarity_maps). Assessable pixels are colored by
departure percentile; the rest are hatched grey ("not assessable"); reference pixels are purple.
Needs outputs/reference_condition_inference/<RUN>_predictors_only_2018 and
<SIMILARITY_DIR>/<RUN>_similarity.tif. Writes <RUN>_assessable_map.png and <RUN>_assessable_summary.csv.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from shapely import affinity

from .make_maps import GREY, INK, MUTED, PURPLE, RAMP, ROOT, block, describe, reference_pixels

NOT_ASSESSABLE = "#b9b7b0"
THRESHOLD = 0.05


def run_one(run: str, sim_dir: Path, out_dir: Path) -> None:
    base, title, _ = describe(run)
    stem = ROOT / "outputs" / "reference_condition_inference" / f"{run}_predictors_only_2018" / f"{run}_predictors_only_2018"
    with rasterio.open(f"{stem}_inference_status.tif") as src:
        target = np.isin(src.read(1), (1, 2))
        transform = src.transform
    with rasterio.open(f"{stem}_reference_departure_percentile.tif") as src:
        departure = src.read(1, masked=True)
    scored = ~np.ma.getmaskarray(departure)
    with rasterio.open(f"{stem}_standardized_deviation.tif") as src:
        reference = reference_pixels(run, transform, ~np.ma.getmaskarray(src.read(1, masked=True)), scored)
    with rasterio.open(sim_dir / f"{run}_similarity.tif") as src:
        similarity = src.read(1)
    inside = np.isfinite(similarity) & (similarity > THRESHOLD)
    km = (abs(transform.a) / 1000) ** 2
    good = scored & inside
    summary = pd.Series({
        "zone_km2": target.sum() * km, "reference_km2": reference.sum() * km,
        "scored_km2": scored.sum() * km, "assessable_scored_km2": good.sum() * km,
        "assessable_share_of_scored": good.sum() / max(scored.sum(), 1),
        "mean_departure_assessable": float(departure[good].mean()) if good.any() else np.nan,
        "mean_departure_not_assessable": float(departure[scored & ~inside].mean()) if (scored & ~inside).any() else np.nan})
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{run}_assessable_summary.csv", header=["value"])
    print(run, summary.round(3).to_dict(), flush=True)

    height, width = target.shape
    f = max(1, int(np.ceil(max(height, width) / 1800)))
    dep_b = block(np.nan_to_num(departure.filled(np.nan)), good, f, "mean")
    na_b = block((target & ~inside).astype(float), target, f, "mean")
    ref_b = block(reference.astype(float), reference, f, "any")
    extent = (transform.c / 1000, (transform.c + width * transform.a) / 1000,
              (transform.f + height * transform.e) / 1000, transform.f / 1000)
    aoi = gpd.read_file(ROOT / "config" / "aoi" / f"{base}.geojson").to_crs("EPSG:6933")
    outline = gpd.GeoSeries([affinity.scale(g, 0.001, 0.001, origin=(0, 0)) for g in aoi.geometry])
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, ax = plt.subplots(figsize=(9.5, min(9.5 * aspect + 1.6, 12)), facecolor="white", layout="constrained")
    outline.plot(ax=ax, facecolor=GREY, edgecolor=MUTED, linewidth=0.7, zorder=0)
    ax.imshow(np.where(na_b >= 0.5, 1.0, np.nan), extent=extent, cmap=ListedColormap([NOT_ASSESSABLE]), interpolation="nearest", zorder=1)
    ax.imshow(np.ma.masked_invalid(dep_b), extent=extent, cmap=RAMP, vmin=0, vmax=1, interpolation="nearest", zorder=2)
    ax.imshow(np.where(ref_b, 1.0, np.nan), extent=extent, cmap=ListedColormap([PURPLE]), interpolation="nearest", zorder=3)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for side in ax.spines.values():
        side.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
    cb.set_label("Departure from modeled reference condition (percentile), assessable pixels only: green = similar, red = far", fontsize=8.5, color=MUTED)
    cb.outline.set_visible(False)
    ax.legend(handles=[Patch(facecolor=PURPLE, label="Reference sites (not scored)"),
                       Patch(facecolor=NOT_ASSESSABLE, label=f"Not assessable: environment outside the reference range ({100 * (1 - summary['assessable_share_of_scored']):.0f}% of scored area)")],
              loc="lower left", frameon=False, fontsize=8.5)
    ax.set_title(f"{title}: where the reference can speak (assessable pixels only)", loc="left", fontsize=12, color=INK)
    fig.savefig(out_dir / f"{run}_assessable_map.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("similarity_dir", type=Path)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for run in args.runs:
        run_one(run, args.similarity_dir, args.output_dir)


if __name__ == "__main__":
    main()
