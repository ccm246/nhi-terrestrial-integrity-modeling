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
import re
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
         "soknot": "SOKNOT", "carpathian": "Carpathian", "sabah": "Sabah", "seka": "SEKA",
         "adirondacks": "Adirondacks", "nepal_tal_2026": "Nepal TAL 2026",
         "adirondacks_large": "Adirondacks (large extent)", "upper_midwest": "Upper Midwest", "guatemala": "Guatemala"}
RAMP = LinearSegmentedColormap.from_list("departure", ["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"])
PURPLE, GREY, INK, MUTED = "#7b3294", "#e9e8e3", "#0b0b0b", "#52514e"
# MODIS IGBP land cover (2018), grouped and muted so the departure ramp stays the focus.
LANDCOVER = {  # group: (color, label, IGBP codes)
    1: ("#c4d3b8", "Forest", (1, 2, 3, 4, 5)),
    2: ("#d9d2b4", "Shrubland / savanna", (6, 7, 8, 9)),
    3: ("#e7e3b0", "Grassland", (10,)),
    4: ("#b9d6cf", "Wetland", (11,)),
    5: ("#ecd9b8", "Cropland / mosaic", (12, 14)),
    6: ("#c9c5c0", "Built-up", (13,)),
    7: ("#e4dccf", "Barren / snow", (15, 16)),
    8: ("#bcd3e4", "Water", (17,)),
}


def describe(run: str) -> tuple[str, str, str]:
    base = re.sub(r"_(w2|forestv2|grassv2|forest|hm|wild)$", "", run)
    kind = "grassland" if run.endswith("_grassv2") else ("forest" if re.search(r"_(forest|forestv2|hm|wild)$", run) else "grassland")
    if run.endswith("_forestv2") or run.endswith("_hm"):
        wave = "forest v2: canopy + human-pressure reference"
    elif run.endswith("_wild"):
        wave = "forest, wilderness-only reference"
    elif run.endswith("_grassv2"):
        wave = "grassland v2: habitat-split reference"
    else:
        wave = "wave 2: selected responses" if "_w2" in run else ("forest reference (IFL + high integrity)" if kind == "forest" else "wave 1: stock workflow")
    return base, f"{NAMES.get(base, base)} {kind}", wave


def true_reference(run: str, transform, shape):
    """Boolean reference-site mask on the inference grid, from the reference raster the run was trained on.

    Returns None when the run's reference raster is not on disk (waves 1 and 2); callers then fall back to
    "complete deviation, not scored", which over-counts wherever ordinary pixels lack some response.
    """
    base = re.sub(r"_(w2|forestv2|grassv2|forest|hm|wild)$", "", run)
    if run.endswith("_forestv2"):
        path = ROOT / "forest_inputs" / "forest_reference_v2" / f"{base}_reference_500m.tif"
    elif run.endswith("_grassv2"):
        path = ROOT / "grass_inputs" / "grass_reference_v2" / f"{base}_reference_500m.tif"
    elif run.endswith("_hm") or run.endswith("_wild"):
        path = ROOT / "transfer_inputs" / base / f"reference_{run.rsplit('_', 1)[1]}_500m.tif"
    elif run.endswith("_forest"):
        path = ROOT / "forest_inputs" / "forest_reference" / f"{base}_reference_500m.tif"
    else:
        return None
    if not path.exists():
        return None
    height, width = shape
    out = np.zeros((height, width), bool)
    with rasterio.open(path) as src:
        data = src.read(1) == 1
        col0 = int(round((src.transform.c - transform.c) / transform.a))
        row0 = int(round((transform.f - src.transform.f) / -transform.e))
    r0, r1 = max(row0, 0), min(row0 + data.shape[0], height)
    c0, c1 = max(col0, 0), min(col0 + data.shape[1], width)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = data[r0 - row0:r1 - row0, c0 - col0:c1 - col0]
    return out


def reference_pixels(run, transform, deviation_mask, scored):
    """True reference sites if the raster exists, else the old approximation."""
    exact = true_reference(run, transform, scored.shape)
    return exact if exact is not None else (deviation_mask & ~scored)


def block(array, valid, factor, how):
    rows, cols = (array.shape[0] // factor) * factor, (array.shape[1] // factor) * factor
    v = np.where(valid, array, 0.0)[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    n = valid[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    count = n.sum(axis=(1, 3))
    if how == "any":
        return count > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, v.sum(axis=(1, 3)) / count, np.nan)


def landcover_groups(base, run, aoi, transform, shape, factor, block_shape):
    """Grouped IGBP land cover subsampled to the map grid and clipped to the AOI (None if not fetched)."""
    from rasterio.features import geometry_mask
    path = ROOT / "data" / "landcover" / f"{run.replace('_w2', '')}_igbp_2018.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        if (src.height, src.width) != shape:
            return None
        cover = src.read(1)
    inside = geometry_mask(aoi.geometry, out_shape=shape, transform=transform, invert=True)
    groups = np.zeros(shape, dtype="uint8")
    for k, (_, _, codes) in LANDCOVER.items():
        groups[np.isin(cover, codes)] = k
    groups[~inside] = 0
    rows, cols = block_shape
    offset = factor // 2
    return groups[offset::factor, offset::factor][:rows, :cols]


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
    reference = reference_pixels(run, transform, ~np.ma.getmaskarray(deviation), scored)
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
    groups = landcover_groups(base, run, aoi, transform, (height, width), factor, p_block.shape)
    if groups is not None:
        ax.imshow(np.ma.masked_equal(groups, 0), extent=extent, interpolation="nearest", zorder=1,
                  cmap=ListedColormap(["white"] + [LANDCOVER[k][0] for k in sorted(LANDCOVER)]), vmin=0, vmax=len(LANDCOVER))
    ax.imshow(np.ma.masked_invalid(p_block), extent=extent, cmap=RAMP, vmin=0, vmax=1, interpolation="nearest", zorder=2)
    ax.imshow(np.where(ref_block, 1.0, np.nan), extent=extent, cmap=ListedColormap([PURPLE]), interpolation="nearest", zorder=3)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for side in ax.spines.values():
        side.set_visible(False)
    if groups is not None:
        from matplotlib.patches import Patch
        present = [k for k in sorted(LANDCOVER) if (groups == k).any()]
        ax.legend(handles=[Patch(facecolor=LANDCOVER[k][0], label=LANDCOVER[k][1]) for k in present] + [Patch(facecolor=PURPLE, label="Reference (not scored)")],
                  title="Land cover (MODIS 2018)", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, title_fontsize=8.5)
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
    runs = args.runs or [f"{n}{t}" for t in ("", "_w2") for n in ("kaza", "pacha", "central_india_boundary", "nepal_tal", "soknot", "carpathian", "sabah")] + ["seka_forest", "sabah_forest", "soknot_forest", "pacha_forest"] + [f"{n}_forestv2" for n in ("adirondacks", "nepal_tal_2026", "seka", "sabah")]
    for run in runs:
        path = make_map(run, args.output_dir)
        print(f"{run:28} {'wrote ' + str(path.name) if path else 'no inference output yet'}", flush=True)


if __name__ == "__main__":
    main()
