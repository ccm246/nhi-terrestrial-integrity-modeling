"""Abiotic similarity-to-reference maps (no vegetation involved).

Usage:
    python -m scripts.tools.make_similarity_maps OUTPUT_DIR RUN [RUN ...]
    e.g. python -m scripts.tools.make_similarity_maps results/maps/similarity adirondacks_forestv2

For every zone pixel in the run's target mask (reference pixels included), computes the
Mahalanobis distance of its continuous climate, soil and terrain predictors to the reference-pixel
distribution (same standardization and 10% covariance shrinkage as flag_extrapolation.py).
Similarity = the share of reference pixels that are farther from the reference center than this
pixel: 1.0 = as typical as the reference core, 0.0 = farther than every reference pixel.
Pixels past the reference's own 95th-percentile distance are "outside the reference environment".

Writes <RUN>_similarity.tif, <RUN>_similarity_summary.csv and <RUN>_similarity_map.png.
Needs the predictor tiles in data/gee_cache_<name>/ and the run's sample parquet.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely import affinity

from ..analysis_config import load_analysis_configuration
from ..raster_cache_utils import resolve_analysis_cache_tiles
from .make_maps import GREY, INK, MUTED, PURPLE, ROOT, block, describe
from .reference_coverage import MIN_COVERAGE, wmean_cov, wquantile

SIM = LinearSegmentedColormap.from_list("similarity", ["#f3e9d2", "#a8cfc3", "#3f9a8d", "#17555e", "#0b2f3a"])
ORANGE = "#e8833a"


def run_one(run: str, out_dir: Path) -> None:
    base, title, wave = describe(run)
    config = load_analysis_configuration(ROOT / "config" / f"{run}_all.toml")
    sample = pd.read_parquet(ROOT / "outputs" / "samples" / f"{run}_all_2018_spatial_sample.parquet")
    folder = ROOT / "outputs" / "reference_condition_inference" / f"{run}_predictors_only_2018"
    stem = f"{run}_predictors_only_2018"
    with rasterio.open(folder / f"{stem}_inference_status.tif") as src:
        status = src.read(1)
        transform, profile = src.transform, src.profile
    target = (status == 1) | (status == 2)   # 255 is nodata
    with rasterio.open(folder / f"{stem}_reference_departure_percentile.tif") as src:
        scored = ~np.ma.getmaskarray(src.read(1, masked=True))
    with rasterio.open(folder / f"{stem}_standardized_deviation.tif") as src:
        reference = ~np.ma.getmaskarray(src.read(1, masked=True)) & ~scored

    with rasterio.open(config.inference.application_mask_path) as mask:
        x, y = Transformer.from_crs("EPSG:4326", mask.crs, always_xy=True).transform(
            sample["longitude"].to_numpy(), sample["latitude"].to_numpy())
        in_mask = np.array([v[0] for v in mask.sample(zip(x, y))]) == 1
    zone = sample[in_mask | (sample["reference_site"] == 1)]
    ref = zone[zone["reference_site"] == 1]
    predictors = config.columns_with_role(sample.columns, "predictor")
    categorical = [b.identifier for b in config.bands if b.role == "predictor" and b.data_type == "categorical"]
    ids = [i for i, c in predictors.items() if i not in categorical
           and zone[c].notna().mean() >= MIN_COVERAGE and ref[c].notna().mean() >= MIN_COVERAGE]
    cols = [predictors[i] for i in ids]
    medians = zone[cols].median()
    G, R = zone[cols].fillna(medians).to_numpy(float), ref[cols].fillna(medians).to_numpy(float)
    w_all, w_ref = zone["area_weight_m2"].to_numpy(), ref["area_weight_m2"].to_numpy()
    center, cov_all = wmean_cov(G, w_all)
    scale = np.sqrt(np.diag(cov_all)); scale[scale == 0] = 1
    r_mean, r_cov = wmean_cov((R - center) / scale, w_ref)
    r_cov = 0.9 * r_cov + 0.1 * np.diag(np.diag(r_cov))
    inverse = np.linalg.pinv(r_cov)

    def distance(matrix):
        z = (matrix - center) / scale - r_mean
        return np.sqrt(np.einsum("ij,jk,ik->i", z, inverse, z))

    d_ref = distance(R)
    limit = wquantile(d_ref, w_ref, 0.95)
    order = np.argsort(d_ref)
    d_sorted, cum = d_ref[order], np.cumsum(w_ref[order]) / w_ref.sum()

    names = list(config.band_names())
    band_index = {i: next(k for k, n in enumerate(names) if f"_{i}_" in n) + 1 for i in ids}
    height, width = status.shape
    stack = np.full((len(ids), height, width), np.nan, "float32")
    for cached in resolve_analysis_cache_tiles(config, show_progress=False).tiles:
        tile = cached.tile
        col0 = int(round((tile.left - transform.c) / transform.a))
        row0 = int(round((transform.f - tile.top) / -transform.e))
        with rasterio.open(cached.path) as src:
            data = src.read([band_index[i] for i in ids], masked=True).astype("float32").filled(np.nan)
        r0, r1 = max(row0, 0), min(row0 + data.shape[1], height)
        c0, c1 = max(col0, 0), min(col0 + data.shape[2], width)
        if r0 < r1 and c0 < c1:
            stack[:, r0:r1, c0:c1] = data[:, r0 - row0:r1 - row0, c0 - col0:c1 - col0]
    pixels = stack[:, target].T.astype(float)
    pixels = np.where(np.isfinite(pixels), pixels, medians.to_numpy()[None, :])
    d = distance(pixels)
    similarity = 1.0 - np.interp(d, d_sorted, cum, left=0.0, right=1.0)   # share of reference farther away
    sim = np.full((height, width), np.nan, "float32")
    sim[target] = similarity
    outside = np.zeros((height, width), bool)
    outside[target] = d > limit

    out_dir.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=np.nan, count=1, compress="zstd")
    with rasterio.open(out_dir / f"{run}_similarity.tif", "w", **profile) as dst:
        dst.write(sim, 1)
    km = (abs(transform.a) / 1000) ** 2
    summary = pd.Series({
        "predictors_used": len(ids), "reference_environments_km2": reference.sum() * km,
        "zone_km2": target.sum() * km, "outside_reference_environment_km2": outside.sum() * km,
        "outside_share": outside[target].mean(),
        "mean_similarity_scored": float(np.nanmean(sim[scored])) if scored.any() else np.nan,
        "mean_similarity_reference": float(np.nanmean(sim[reference])) if reference.any() else np.nan})
    summary.to_csv(out_dir / f"{run}_similarity_summary.csv", header=["value"])
    print(run, summary.round(3).to_dict(), flush=True)

    f = max(1, int(np.ceil(max(height, width) / 1800)))
    sim_b = block(np.nan_to_num(sim), target, f, "mean")
    out_b = block(outside.astype(float), target, f, "mean")
    ref_b = block(reference.astype(float), reference, f, "any")
    extent = (transform.c / 1000, (transform.c + width * transform.a) / 1000,
              (transform.f + height * transform.e) / 1000, transform.f / 1000)
    aoi = gpd.read_file(ROOT / "config" / "aoi" / f"{base}.geojson").to_crs("EPSG:6933")
    outline = gpd.GeoSeries([affinity.scale(g, 0.001, 0.001, origin=(0, 0)) for g in aoi.geometry])
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, min(7.5 * aspect + 1.8, 12)), facecolor="white")
    for ax, which in zip(axes, ("sim", "outside")):
        outline.plot(ax=ax, facecolor=GREY, edgecolor=MUTED, linewidth=0.7, zorder=0)
        ax.imshow(np.ma.masked_invalid(sim_b), extent=extent, cmap=SIM, vmin=0, vmax=1, interpolation="nearest", zorder=2)
        if which == "outside":
            ax.imshow(np.where(out_b >= 0.5, 1.0, np.nan), extent=extent, cmap=ListedColormap([ORANGE]), interpolation="nearest", zorder=3)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_title("Environmental similarity to the reference", fontsize=12, loc="left", color=INK)
    axes[1].set_title("Environments outside the reference range", fontsize=12, loc="left", color=INK)
    sm = plt.cm.ScalarMappable(cmap=SIM, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=list(axes), orientation="horizontal", fraction=0.04, pad=0.03, aspect=60)
    cb.set_label("Share of reference pixels farther from the reference core (1 = typical of reference)", fontsize=8.5, color=MUTED)
    cb.outline.set_visible(False)
    axes[1].legend(handles=[Patch(facecolor=ORANGE, label=f"Outside reference range: {100 * summary['outside_share']:.0f}% of the scored zone")],
                   loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(f"{title}: abiotic similarity to reference (climate, soil and terrain only)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(out_dir / f"{run}_similarity_map.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for run in args.runs:
        run_one(run, args.output_dir)


if __name__ == "__main__":
    main()
