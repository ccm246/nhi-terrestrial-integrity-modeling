"""Flag scored grassland pixels whose environment lies outside the reference envelope.

Usage:
    python -m scripts.tools.flag_extrapolation RUN_NAME

For every pixel scored by the predictors-only integrity map, computes the Mahalanobis
distance of its continuous environmental predictors to the reference-pixel
distribution (10 % covariance shrinkage, as in the inference step). Pixels farther
than the reference pixels' own weighted 95th-percentile distance are flagged: their
departure scores extrapolate beyond the environments the reference pixels cover.

Writes outputs/reference_coverage/<RUN>_extrapolation_flag.tif (0 = inside,
1 = outside, 255 = not scored), a summary CSV, and a two-panel map figure.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap

from ..analysis_config import load_analysis_configuration
from ..raster_cache_utils import resolve_analysis_cache_tiles
from .reference_coverage import MIN_COVERAGE, wmean_cov, wquantile
from .reference_coverage import BLUE, ORANGE
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
RAMP = LinearSegmentedColormap.from_list("departure", ["#e8f1fb", "#9cc3ee", "#2a78d6", "#0b2f5e"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    config = load_analysis_configuration(ROOT / "config" / f"{args.run}_all.toml")
    sample = pd.read_parquet(ROOT / "outputs" / "samples" / f"{args.run}_all_2018_spatial_sample.parquet")
    inference = (ROOT / "outputs" / "reference_condition_inference"
                 / f"{args.run}_predictors_only_2018")
    stem = f"{args.run}_predictors_only_2018"
    with rasterio.open(inference / f"{stem}_reference_departure_percentile.tif") as src:
        percentile = src.read(1, masked=True).astype("float32")
        transform, profile = src.transform, src.profile
    scored = ~np.ma.getmaskarray(percentile)

    # Environments used by the coverage analysis: grassland-mask sample rows + reference.
    with rasterio.open(ROOT / "data" / "grassland_masks" / f"{args.run}_all_grassland_mask_2018.tif") as mask:
        x, y = Transformer.from_crs("EPSG:4326", mask.crs, always_xy=True).transform(
            sample["longitude"].to_numpy(), sample["latitude"].to_numpy())
        in_mask = np.array([v[0] for v in mask.sample(zip(x, y))]) == 1
    grass = sample[in_mask | (sample["reference_site"] == 1)]
    reference = grass[grass["reference_site"] == 1]
    predictors = config.columns_with_role(sample.columns, "predictor")
    categorical = next(b.identifier for b in config.bands
                       if b.role == "predictor" and b.data_type == "categorical")
    ids = [i for i, c in predictors.items() if i != categorical
           and grass[c].notna().mean() >= MIN_COVERAGE and reference[c].notna().mean() >= MIN_COVERAGE]
    cols = [predictors[i] for i in ids]
    medians = grass[cols].median()
    weights_all, weights_ref = grass["area_weight_m2"].to_numpy(), reference["area_weight_m2"].to_numpy()
    G, R = grass[cols].fillna(medians).to_numpy(float), reference[cols].fillna(medians).to_numpy(float)
    center, cov_all = wmean_cov(G, weights_all)
    scale = np.sqrt(np.diag(cov_all)); scale[scale == 0] = 1
    r_mean, r_cov = wmean_cov((R - center) / scale, weights_ref)
    r_cov = 0.9 * r_cov + 0.1 * np.diag(np.diag(r_cov))
    inverse = np.linalg.pinv(r_cov)

    def distance(matrix):
        z = (matrix - center) / scale - r_mean
        return np.sqrt(np.einsum("ij,jk,ik->i", z, inverse, z))

    limit = wquantile(distance(R), weights_ref, 0.95)

    # Predictor stack for every scored pixel, read from the cached tiles.
    names = list(config.band_names())
    band_index = {i: next(k for k, n in enumerate(names) if f"_{i}_" in n) + 1 for i in ids}
    height, width = percentile.shape
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
    pixels = stack[:, scored].T.astype(float)
    pixels = np.where(np.isfinite(pixels), pixels, medians.to_numpy()[None, :] * 0 + medians.to_numpy())
    far = distance(pixels) > limit
    flag = np.full((height, width), 255, "uint8")
    flag[scored] = far.astype("uint8")

    out = ROOT / "outputs" / "reference_coverage"
    profile.update(dtype="uint8", nodata=255, count=1, compress="zstd")
    with rasterio.open(out / f"{args.run}_extrapolation_flag.tif", "w", **profile) as dst:
        dst.write(flag, 1)
    pct = percentile[scored].filled(np.nan)
    summary = pd.Series({
        "scored_pixels": int(scored.sum()),
        "scored_km2": scored.sum() * 0.25,
        "flagged_share": far.mean(),
        "flagged_km2": far.sum() * 0.25,
        "mean_departure_inside": float(np.nanmean(pct[~far])),
        "mean_departure_flagged": float(np.nanmean(pct[far])),
    })
    summary.to_csv(out / f"{args.run}_extrapolation_summary.csv", header=["value"])
    print(summary.round(3).to_string())

    # Figure: departure map, plus the same with extrapolated pixels highlighted.
    block = 2
    rows, cols_ = (height // block) * block, (width // block) * block
    def reduce(values, valid):
        v = np.where(valid, values, 0.0)[:rows, :cols_].reshape(rows // block, block, cols_ // block, block)
        n = valid[:rows, :cols_].reshape(rows // block, block, cols_ // block, block)
        count = n.sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, v.sum(axis=(1, 3)) / count, np.nan)
    departure = reduce(percentile.filled(0), scored)
    flagged = reduce((flag == 1).astype(float), scored)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="white")
    extent = (transform.c / 1000, (transform.c + width * 500) / 1000,
              (transform.f - height * 500) / 1000, transform.f / 1000)
    import geopandas as gpd
    from shapely import affinity
    aoi = gpd.read_file(config.aoi_path).to_crs("EPSG:6933")
    outline = gpd.GeoSeries([affinity.scale(g, 0.001, 0.001, origin=(0, 0)) for g in aoi.geometry])
    for ax, title, highlight in zip(axes, ("Departure from reference (all scored grassland)",
                                           "Same map, extrapolated pixels highlighted"), (False, True)):
        outline.plot(ax=ax, facecolor="#ecebe6", edgecolor="#8d8c85", linewidth=0.8, zorder=0)
        ax.imshow(departure, extent=extent, cmap=RAMP, vmin=0, vmax=1, interpolation="nearest", zorder=2)
        if highlight:
            overlay = np.where(flagged > 0, 1.0, np.nan)
            ax.imshow(overlay, extent=extent, cmap=LinearSegmentedColormap.from_list("o", [ORANGE, ORANGE]),
                      interpolation="nearest", zorder=3)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
        for side in ax.spines.values():
            side.set_visible(False)
    fig.text(0.52, 0.02,
                 f"Orange: {100 * far.mean():.0f}% of scored grassland ({far.sum() * 0.25:,.0f} km²) has an\n"
                 "environment outside the reference pixels' envelope;\nits score extrapolates and is less reliable.",
                 fontsize=10, va="bottom")
    fig.suptitle(f"{config.display_name}: reliability of departure scores", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out / f"{args.run}_extrapolation_map.png", dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
