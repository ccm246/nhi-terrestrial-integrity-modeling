"""Reference-only, scored-extent, Adirondack land-class, and observed/predicted/residual maps.

Usage:
    python -m scripts.tools.make_diagnostic_maps extent OUTPUT_DIR [RUN ...]
    python -m scripts.tools.make_diagnostic_maps landclass OUTPUT_DIR
    python -m scripts.tools.make_diagnostic_maps residual OUTPUT_DIR [RUN ...]

extent    writes <run>_reference_map.png (reference pixels only, on land cover) and
          <run>_extent_map.png (reference / scored / not scored inside the outline).
landclass writes adirondacks_landclass_map.png (Park Agency classes beside departure).
residual  writes <run>_obs_pred_resid.png: observed, model-predicted (expected under reference
          conditions) and observed-minus-expected for the best-fitting response, NDVI median
          (d03) and the weakest response. R2 is the held-out R2 from response_metrics.csv.

Reference pixels are those with a complete response set that were not scored, as in make_maps.
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
from rasterio.features import geometry_mask, rasterize
from shapely import affinity

from .make_maps import (GREY, INK, LANDCOVER, MUTED, PURPLE, RAMP, ROOT, block, describe, landcover_groups, reference_pixels)

TEAL = "#2f8f83"
WAVE1 = ["kaza", "pacha", "central_india_boundary", "nepal_tal", "soknot", "carpathian", "sabah"]
FOREST = [f"{n}_forestv2" for n in ("adirondacks", "nepal_tal_2026", "seka", "sabah")]


def load(run, max_px=1800):
    folder = ROOT / "outputs" / "reference_condition_inference" / f"{run}_predictors_only_2018"
    stem = f"{run}_predictors_only_2018"
    if not (folder / f"{stem}_reference_departure_percentile.tif").exists():
        return None
    d = {}
    with rasterio.open(folder / f"{stem}_reference_departure_percentile.tif") as s:
        pct = s.read(1, masked=True); d["transform"] = s.transform
    with rasterio.open(folder / f"{stem}_standardized_deviation.tif") as s:
        dev = s.read(1, masked=True)
    with rasterio.open(folder / f"{stem}_inference_status.tif") as s:
        status = s.read(1, masked=True)
    d["scored"] = ~np.ma.getmaskarray(pct)
    d["reference"] = reference_pixels(run, d["transform"], ~np.ma.getmaskarray(dev), d["scored"])
    d["target"] = ~np.ma.getmaskarray(status) & (status.filled(0) > 0)
    d["pct"] = pct
    d["folder"], d["stem"] = folder, stem
    h, w = pct.shape
    d["shape"] = (h, w)
    d["factor"] = max(1, int(np.ceil(max(h, w) / max_px)))
    t = d["transform"]
    d["extent"] = (t.c / 1000, (t.c + w * t.a) / 1000, (t.f + h * t.e) / 1000, t.f / 1000)
    d["px_km2"] = (abs(t.a) / 1000) ** 2
    return d


def outline_for(base):
    aoi = gpd.read_file(ROOT / "config" / "aoi" / f"{base}.geojson").to_crs("EPSG:6933")
    return aoi, gpd.GeoSeries([affinity.scale(g, 0.001, 0.001, origin=(0, 0)) for g in aoi.geometry])


def frame(d, outline, fig_w=9.5, extra=2.0):
    e = d["extent"]
    aspect = (e[3] - e[2]) / (e[1] - e[0])
    fig, ax = plt.subplots(figsize=(fig_w, min(fig_w * aspect + extra, 13)), facecolor="white")
    outline.plot(ax=ax, facecolor=GREY, edgecolor=MUTED, linewidth=0.7, zorder=0)
    return fig, ax


def finish(ax, d):
    e = d["extent"]
    ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for side in ax.spines.values():
        side.set_visible(False)


def extent_maps(run, out_dir):
    d = load(run)
    if d is None:
        return None
    base, title, wave = describe(run)
    aoi, outline = outline_for(base)
    f, shape = d["factor"], d["shape"]
    km = d["px_km2"]
    ref_b = block(d["reference"].astype(float), d["reference"], f, "any")
    scored_b = block(d["scored"].astype(float), d["scored"], f, "any")
    aoi_km2 = aoi.geometry.area.sum() / 1e6
    ref_km2, scored_km2 = d["reference"].sum() * km, d["scored"].sum() * km
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = frame(d, outline)
    groups = landcover_groups(base, run, aoi, d["transform"], shape, f, ref_b.shape)
    if groups is not None:
        ax.imshow(np.ma.masked_equal(groups, 0), extent=d["extent"], interpolation="nearest", zorder=1,
                  cmap=ListedColormap(["white"] + [LANDCOVER[k][0] for k in sorted(LANDCOVER)]), vmin=0, vmax=len(LANDCOVER))
    ax.imshow(np.where(ref_b, 1.0, np.nan), extent=d["extent"], cmap=ListedColormap([PURPLE]), interpolation="nearest", zorder=3)
    finish(ax, d)
    handles = [Patch(facecolor=PURPLE, label="Reference pixels")]
    if groups is not None:
        handles += [Patch(facecolor=LANDCOVER[k][0], label=LANDCOVER[k][1]) for k in sorted(LANDCOVER) if (groups == k).any()]
    ax.legend(handles=handles, title="Land cover (MODIS 2018)", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, title_fontsize=8.5)
    ax.set_title(f"{title}: reference pixels only\n{wave}", fontsize=13, loc="left", color=INK)
    fig.text(0.02, 0.012, f"Reference {ref_km2:,.0f} km² = {100 * ref_km2 / aoi_km2:.1f}% of the {aoi_km2:,.0f} km² outline", fontsize=8.5, color=MUTED)
    p1 = out_dir / f"{run}_reference_map.png"
    fig.savefig(p1, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)

    fig, ax = frame(d, outline)
    ax.imshow(np.where(scored_b, 1.0, np.nan), extent=d["extent"], cmap=ListedColormap([TEAL]), interpolation="nearest", zorder=2)
    ax.imshow(np.where(ref_b, 1.0, np.nan), extent=d["extent"], cmap=ListedColormap([PURPLE]), interpolation="nearest", zorder=3)
    finish(ax, d)
    ax.legend(handles=[Patch(facecolor=TEAL, label=f"Scored ({scored_km2:,.0f} km²)"),
                       Patch(facecolor=PURPLE, label=f"Reference, not scored ({ref_km2:,.0f} km²)"),
                       Patch(facecolor=GREY, label="Inside outline, not modeled")],
              loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8.5)
    ax.set_title(f"{title}: scored extent\n{wave}", fontsize=13, loc="left", color=INK)
    fig.text(0.02, 0.012, f"Modeled {100 * (scored_km2 + ref_km2) / aoi_km2:.0f}% of the {aoi_km2:,.0f} km² outline; "
             f"the rest is outside the grassland or forest mask, or lacks data", fontsize=8.5, color=MUTED)
    p2 = out_dir / f"{run}_extent_map.png"
    fig.savefig(p2, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return p1, p2


# --- Adirondack land classes -------------------------------------------------------------
STATE = {7: "Wilderness", 8: "Canoe Area", 9: "Primitive", 10: "Wild Forest", 11: "Intensive Use",
         12: "Historic", 13: "State Administrative", 14: "Pending Classification"}
PRIVATE = {1: "Hamlet", 2: "Moderate Intensity", 3: "Low Intensity", 4: "Rural Use",
           5: "Resource Management", 6: "Industrial Use", 0: "Pending Private"}
GROUPS = [  # (label, color, [(layer, code)...])
    ("State: Wilderness / Primitive / Canoe", "#1b5e3b", [(1, 7), (1, 8), (1, 9)]),
    ("State: Wild Forest", "#6aa84f", [(1, 10)]),
    ("State: Intensive Use / other", "#8a6bb0", [(1, 11), (1, 12), (1, 13), (1, 14)]),
    ("Private: Resource Management", "#c5d18a", [(0, 5)]),
    ("Private: Rural Use", "#e6d58b", [(0, 4)]),
    ("Private: Low / Moderate Intensity", "#e8a65f", [(0, 3), (0, 2)]),
    ("Private: Hamlet / Industrial", "#a23b3b", [(0, 1), (0, 6)]),
    ("Private: Pending", "#bdbdbd", [(0, 0)]),
]


def landclass_map(out_dir):
    d = load("adirondacks_forestv2", max_px=900)
    aoi, outline = outline_for("adirondacks")
    shape, tr = d["shape"], d["transform"]
    grid = np.zeros(shape, dtype="uint8")
    for layer in (0, 1):  # private first, state overrides
        frame_gdf = gpd.read_file(ROOT / "data" / "external" / "adirondacks" / f"adirondack_landclass_layer{layer}.geojson").to_crs("EPSG:6933")
        for k, (_, _, members) in enumerate(GROUPS, start=1):
            for lyr, code in members:
                if lyr != layer:
                    continue
                shapes = [(g, 1) for g in frame_gdf.loc[frame_gdf["LCCode"] == code, "geometry"]]
                if shapes:
                    grid[rasterize(shapes, out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)] = k
    inside = geometry_mask(aoi.geometry, out_shape=shape, transform=tr, invert=True)
    grid[~inside] = 0
    e = d["extent"]
    aspect = (e[3] - e[2]) / (e[1] - e[0])
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2 * aspect + 2.2), facecolor="white")
    pct_b = block(d["pct"].filled(0), d["scored"], d["factor"], "mean")
    ref_b = block(d["reference"].astype(float), d["reference"], d["factor"], "any")
    for ax in axes:
        outline.plot(ax=ax, facecolor=GREY, edgecolor=MUTED, linewidth=0.7, zorder=0)
    f = d["factor"]
    off = f // 2
    cls = grid[off::f, off::f][:pct_b.shape[0], :pct_b.shape[1]]
    axes[0].imshow(np.ma.masked_equal(cls, 0), extent=e, interpolation="nearest", zorder=2,
                   cmap=ListedColormap([g[1] for g in GROUPS]), vmin=1, vmax=len(GROUPS))
    axes[1].imshow(np.ma.masked_invalid(pct_b), extent=e, cmap=RAMP, vmin=0, vmax=1, interpolation="nearest", zorder=2)
    axes[1].imshow(np.where(ref_b, 1.0, np.nan), extent=e, cmap=ListedColormap([PURPLE]), interpolation="nearest", zorder=3)
    for ax in axes:
        finish(ax, d)
    axes[0].set_title("Adirondack Park Agency land classification", fontsize=12, loc="left", color=INK)
    axes[1].set_title("Departure from modeled forest reference (forest v2)", fontsize=12, loc="left", color=INK)
    axes[0].legend(handles=[Patch(facecolor=g[1], label=g[0]) for g in GROUPS], loc="upper center",
                   bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=8.5)
    axes[1].legend(handles=[Patch(facecolor=PURPLE, label="Reference (not scored)"), Patch(facecolor="#1a9850", label="Close to reference"),
                            Patch(facecolor="#a50026", label="Far from reference")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=8.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adirondacks_landclass_map.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return path


# --- observed / predicted / residual -----------------------------------------------------
def residual_map(run, out_dir):
    d = load(run, max_px=700)
    if d is None:
        return None
    base, title, wave = describe(run)
    _, outline = outline_for(base)
    metrics = pd.read_csv(ROOT / "outputs" / "integrity_parameters" / f"{run}_predictors_only" / "response_metrics.csv")
    metrics = metrics.dropna(subset=["overall_weighted_r2"])
    with rasterio.open(d["folder"] / f"{d['stem']}_expected_reference.tif") as s:
        expected, names = s.read(masked=True), list(s.descriptions)
    with rasterio.open(d["folder"] / f"{d['stem']}_observed_minus_expected.tif") as s:
        resid = s.read(masked=True)
    code_to_band = {n.split("_")[0]: i for i, n in enumerate(names)}
    avail = metrics[metrics["response_band"].isin(code_to_band)].sort_values("overall_weighted_r2")
    picks = [avail.iloc[-1]["response_band"]]
    if "d03" in code_to_band and "d03" not in picks:
        picks.append("d03")
    if avail.iloc[0]["response_band"] not in picks:
        picks.append(avail.iloc[0]["response_band"])
    e = d["extent"]
    aspect = (e[3] - e[2]) / (e[1] - e[0])
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, len(picks) * (3.6 * aspect + 0.9) + 0.8), facecolor="white", squeeze=False)
    f = d["factor"]
    valid = ~np.ma.getmaskarray(expected[0])
    for row, band in enumerate(picks):
        i = code_to_band[band]
        exp = expected[i].filled(np.nan); res = resid[i].filled(np.nan)
        obs = exp + res
        info = metrics.loc[metrics["response_band"] == band].iloc[0]
        ok = np.isfinite(exp) & np.isfinite(res)
        lo, hi = np.nanpercentile(np.concatenate([obs[ok], exp[ok]]), [2, 98])
        lim = np.nanpercentile(np.abs(res[ok]), 98)
        panels = [("Observed", obs, "viridis", lo, hi), ("Predicted (expected under reference)", exp, "viridis", lo, hi),
                  ("Observed minus predicted", res, "RdBu_r", -lim, lim)]
        for col, (name, arr, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row][col]
            outline.plot(ax=ax, facecolor=GREY, edgecolor=MUTED, linewidth=0.5, zorder=0)
            b = block(np.where(ok, arr, 0), ok, f, "mean")
            im = ax.imshow(np.ma.masked_invalid(b), extent=e, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", zorder=2)
            ax.set_xlim(e[0], e[1]); ax.set_ylim(e[2], e[3]); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
            for side in ax.spines.values():
                side.set_visible(False)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, aspect=30)
            cb.ax.tick_params(labelsize=7); cb.ax.locator_params(nbins=4); cb.outline.set_visible(False)
            ax.set_title(f"{info['display_name']} ({band}): {name}" if col == 0 else name, fontsize=9.5, loc="left", color=INK)
        axes[row][0].set_ylabel(f"held-out R² = {info['overall_weighted_r2']:.2f}", fontsize=9, color=MUTED)
    fig.suptitle(f"{title}: observed, predicted and residual ({wave})", fontsize=13, x=0.01, ha="left", color=INK)
    fig.text(0.01, 0.003, "Predicted = what the model expects from climate, soil and terrain alone, trained on reference pixels. "
             "Positive residual = observed higher than expected. Rows: best-fitting response, NDVI median, weakest response.",
             fontsize=8, color=MUTED)
    fig.subplots_adjust(hspace=0.25, wspace=0.08)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run}_obs_pred_resid.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["extent", "landclass", "residual"])
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()
    if args.kind == "landclass":
        print(landclass_map(args.output_dir))
        return
    for run in args.runs or WAVE1 + FOREST:
        result = (extent_maps if args.kind == "extent" else residual_map)(run, args.output_dir)
        print(f"{run:28} {'ok' if result else 'no inference output yet'}", flush=True)


if __name__ == "__main__":
    main()
