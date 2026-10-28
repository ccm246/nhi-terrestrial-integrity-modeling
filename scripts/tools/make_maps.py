"""Integrity maps for every finished run (departure from the modeled reference condition).

Usage:
    python -m scripts.tools.make_maps OUTPUT_DIR [RUN ...]

RUN is the inference folder name without the trailing ``_predictors_only_2018``
(e.g. ``kaza``, ``kaza_w2``, ``sabah_forest``). Each map shows the landscape outline, the
reference calibration pixels (purple, not scored), and every scored pixel colored by its
departure percentile: green = within the common range of reference variation, red = far
from it. Pixels are aggregated to <= ~1,800 px on the long side (reference wins, otherwise the
mean departure), so small patches stay visible.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from shapely import affinity

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"kaza": "KAZA", "pacha": "PACHA", "central_india_boundary": "Central India", "nepal_tal": "Nepal TAL",
         "soknot": "SOKNOT", "carpathian": "Carpathian", "sabah": "Sabah", "seka": "SEKA"}
RAMP = LinearSegmentedColormap.from_list("departure", ["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"])
PURPLE, GREY, INK, MUTED = "#7b3294", "#e9e8e3", "#0b0b0b", "#52514e"


def describe(run: str) -> tuple[str, str, str]:
    base = run.replace("_w2", "").replace("_forest", "")
    kind = "forest" if run.endswith("_forest") else "grassland"
    wave = "wave 2: selected responses" if "_w2" in run else ("forest reference (IFL + high integrity)" if kind == "forest" else "wave 1: stock workflow")
    return base, f"{NAMES.get(base, base)} {kind}", wave


def block(array, valid, factor, how):
    rows, cols = (array.shape[0] // factor) * factor, (array.shape[1] // factor) * factor
    v = np.where(valid, array, 0.0)[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    n = valid[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    count = n.sum(axis=(1, 3))
    if how == "any":
        return count > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, v.sum(axis=(1, 3)) / count, np.nan)


def make_map(run: str, out_dir: Path) -> Path | None:
    folder = ROOT / "outputs" / "reference_condition_inference" / f"{run}_predictors_only_2018"
    stem = f"{run}_predictors_only_2018"
    files = {k: folder / f"{stem}_{k}.tif" for k in ("reference_departure_percentile", "standardized_deviation", "inference_status")}
    if not all(p.exists() for p in files.values()):
        return None
    base, title, wave = describe(run)
    with rasterio.open(files["reference_departure_percentile"]) as src:
        percentile = src.read(1, masked=True); transform = src.transform
    with rasterio.open(files["standardized_deviation"]) as src:
        deviation = src.read(1, masked=True)
    with rasterio.open(files["inference_status"]) as src:
        status = src.read(1, masked=True)
    scored = ~np.ma.getmaskarray(percentile)
    reference = ~np.ma.getmaskarray(deviation) & ~scored          # complete response, not scored = reference
    in_target = ~np.ma.getmaskarray(status) & (status.filled(0) > 0)
    height, width = percentile.shape
    factor = max(1, int(np.ceil(max(height, width) / 1800)))
    p_block = block(percentile.filled(0), scored, factor, "mean")
    ref_block = block(reference.astype(float), reference, factor, "any")
    px_km2 = (abs(transform.a) / 1000) ** 2
    values = percentile.compressed()
    stats = (f"{scored.sum() * px_km2:,.0f} km² scored | reference {reference.sum() * px_km2:,.0f} km² (purple) | "
             f"mean departure {values.mean():.2f} | {100 * (values >= 0.95).mean():.0f}% beyond the 95th percentile")
    extent = (transform.c / 1000, (transform.c + width * transform.a) / 1000,
              (transform.f + height * transform.e) / 1000, transform.f / 1000)
    aoi = gpd.read_file(ROOT / "config" / "aoi" / f"{base}.geojson").to_crs("EPSG:6933")
    outline = gpd.GeoSeries([affinity.scale(g, 0.001, 0.001, origin=(0, 0)) for g in aoi.geometry])
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig_w = 9.5
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_w * aspect + 2.0, 13)), facecolor="white")
    outline.plot(ax=ax, facecolor=GREY, edgecolor=MUTED, linewidth=0.7, zorder=0)
    ax.imshow(np.ma.masked_invalid(p_block), extent=extent, cmap=RAMP, vmin=0, vmax=1, interpolation="nearest", zorder=2)
    ax.imshow(np.where(ref_block, 1.0, np.nan), extent=extent, cmap=ListedColormap([PURPLE]), interpolation="nearest", zorder=3)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{title}\n{wave}", fontsize=13, loc="left", color=INK)
    sm = plt.cm.ScalarMappable(cmap=RAMP, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.03, aspect=40)
    cbar.set_label("Departure from modeled reference condition (percentile): green = similar, red = far", fontsize=9, color=MUTED)
    cbar.outline.set_visible(False)
    fig.text(0.02, 0.012, stats, fontsize=8.5, color=MUTED)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run}_map.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()
    runs = args.runs or [f"{n}{t}" for t in ("", "_w2") for n in ("kaza", "pacha", "central_india_boundary", "nepal_tal", "soknot", "carpathian", "sabah")] + ["seka_forest", "sabah_forest", "soknot_forest", "pacha_forest"]
    for run in runs:
        path = make_map(run, args.output_dir)
        print(f"{run:28} {'wrote ' + str(path.name) if path else 'no inference output yet'}", flush=True)


if __name__ == "__main__":
    main()
