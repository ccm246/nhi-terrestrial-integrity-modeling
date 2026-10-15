"""How much of a landscape's natural grassland do the reference pixels represent?

Usage:
    python -m scripts.tools.reference_coverage RUN_NAME

RUN_NAME (e.g. ``central_india_boundary``) selects its config, spatial sample and
grassland mask. "Grassland" is every sampled pixel inside the landscape's grassland
mask (area-weighted); "reference" is the reference-site class. For each continuous
environmental predictor and for the predictors jointly, it reports how much of the
grassland falls inside the reference pixels' environmental range.

Writes outputs/reference_coverage/<RUN_NAME>_*.csv and *.png.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from ..analysis_config import load_analysis_configuration

ROOT = Path(__file__).resolve().parents[2]
BLUE, GREY, ORANGE = "#2a78d6", "#8d8c85", "#eb6834"
MIN_COVERAGE = 0.8


def wquantile(values, weights, quantiles):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(quantiles, cumulative, v)


def wmean_cov(matrix, weights):
    w = weights / weights.sum()
    mean = w @ matrix
    centered = matrix - mean
    return mean, (centered * w[:, None]).T @ centered


def histogram_overlap(a, wa, b, wb, bins):
    ha, _ = np.histogram(a, bins=bins, weights=wa)
    hb, _ = np.histogram(b, bins=bins, weights=wb)
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    args = parser.parse_args()
    config = load_analysis_configuration(ROOT / "config" / f"{args.run}_all.toml")
    sample = pd.read_parquet(
        ROOT / "outputs" / "samples" / f"{args.run}_all_2018_spatial_sample.parquet"
    )
    mask_path = ROOT / "data" / "grassland_masks" / f"{args.run}_all_grassland_mask_2018.tif"
    with rasterio.open(mask_path) as mask:
        x, y = Transformer.from_crs("EPSG:4326", mask.crs, always_xy=True).transform(
            sample["longitude"].to_numpy(), sample["latitude"].to_numpy()
        )
        values = np.fromiter(mask.sample(zip(x, y), indexes=1, masked=False), dtype="uint8",
                             count=len(sample)) if False else np.array(
            [v[0] for v in mask.sample(zip(x, y), indexes=1)], dtype="uint8")
    sample["in_grassland"] = values == 1
    reference_flag = sample["reference_site"] == 1
    grass = sample[sample["in_grassland"] | reference_flag].copy()  # reference counts as grassland
    reference = grass[grass["reference_site"] == 1]
    weight_all, weight_ref = grass["area_weight_m2"].to_numpy(), reference["area_weight_m2"].to_numpy()
    print(f"grassland rows {len(grass):,} ({weight_all.sum() / 1e6:,.0f} km2); "
          f"reference rows {len(reference):,} ({weight_ref.sum() / 1e6:,.0f} km2, "
          f"{100 * weight_ref.sum() / weight_all.sum():.1f}% of grassland)")

    names = {b.identifier: b.display_name for b in config.bands}
    columns = config.columns_with_role(sample.columns, "predictor")
    categorical = next(b.identifier for b in config.bands
                       if b.role == "predictor" and b.data_type == "categorical")
    continuous = {}
    for identifier, column in columns.items():
        if identifier == categorical:
            continue
        if grass[column].notna().mean() >= MIN_COVERAGE and reference[column].notna().mean() >= MIN_COVERAGE:
            continuous[identifier] = column

    records = []
    for identifier, column in continuous.items():
        g, r = grass[column], reference[column]
        gv, gw = g[g.notna()].to_numpy(), grass.loc[g.notna(), "area_weight_m2"].to_numpy()
        rv, rw = r[r.notna()].to_numpy(), reference.loc[r.notna(), "area_weight_m2"].to_numpy()
        lo5, hi95 = wquantile(rv, rw, [0.05, 0.95])
        in_minmax = np.average((gv >= rv.min()) & (gv <= rv.max()), weights=gw)
        in_90 = np.average((gv >= lo5) & (gv <= hi95), weights=gw)
        gm, gs = np.average(gv, weights=gw), np.sqrt(np.cov(gv, aweights=gw))
        bins = np.linspace(min(gv.min(), rv.min()), max(gv.max(), rv.max()), 31)
        records.append({
            "band": identifier, "variable": names[identifier],
            "grassland_median": wquantile(gv, gw, 0.5), "reference_median": wquantile(rv, rw, 0.5),
            "std_mean_difference": (np.average(rv, weights=rw) - gm) / gs,
            "histogram_overlap": histogram_overlap(gv, gw, rv, rw, bins),
            "grassland_inside_reference_min_max": in_minmax,
            "grassland_inside_reference_5_95": in_90,
        })
    table = pd.DataFrame(records)

    # Joint (multivariate) coverage: Mahalanobis distance to the reference distribution.
    cols = list(continuous.values())
    medians = grass[cols].median()
    G = grass[cols].fillna(medians).to_numpy(float)
    R = reference[cols].fillna(medians).to_numpy(float)
    scale = np.sqrt(np.diag(wmean_cov(G, weight_all)[1]))
    scale[scale == 0] = 1
    G, R = (G - wmean_cov(G, weight_all)[0]) / scale, (R - wmean_cov(G, weight_all)[0]) / scale
    r_mean, r_cov = wmean_cov(R, weight_ref)
    shrink = 0.1
    r_cov = (1 - shrink) * r_cov + shrink * np.diag(np.diag(r_cov))
    inverse = np.linalg.pinv(r_cov)
    distance = lambda M: np.sqrt(np.einsum("ij,jk,ik->i", M - r_mean, inverse, M - r_mean))
    d_ref, d_all = distance(R), distance(G)
    limit95 = wquantile(d_ref, weight_ref, 0.95)
    joint_inside = float(np.average(d_all <= limit95, weights=weight_all))
    outside_any = float(np.average(
        ((grass[cols].to_numpy(float) < reference[cols].min().to_numpy())
         | (grass[cols].to_numpy(float) > reference[cols].max().to_numpy())).any(axis=1),
        weights=weight_all))

    # Landform shares.
    lf = columns[categorical]
    shares = pd.DataFrame({
        "grassland": grass.groupby(lf)["area_weight_m2"].sum() / weight_all.sum(),
        "reference": reference.groupby(lf)["area_weight_m2"].sum() / weight_ref.sum(),
    }).fillna(0)

    out = ROOT / "outputs" / "reference_coverage"
    out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / f"{args.run}_variable_coverage.csv", index=False)
    shares.round(4).to_csv(out / f"{args.run}_landform_shares.csv")
    summary = {
        "grassland_km2": weight_all.sum() / 1e6, "reference_km2": weight_ref.sum() / 1e6,
        "reference_share_of_grassland": weight_ref.sum() / weight_all.sum(),
        "grassland_inside_joint_95_envelope": joint_inside,
        "grassland_outside_range_of_any_variable": outside_any,
        "median_var_inside_5_95": table["grassland_inside_reference_5_95"].median(),
    }
    pd.Series(summary).to_csv(out / f"{args.run}_summary.csv", header=["value"])
    print(pd.Series(summary).round(3).to_string())
    print(table.sort_values("grassland_inside_reference_5_95")[
        ["variable", "std_mean_difference", "histogram_overlap",
         "grassland_inside_reference_5_95", "grassland_inside_reference_min_max"]].round(2).to_string(index=False))
    print(shares.round(3).to_string())

    # Figure 1: per-variable distributions.
    ids = list(continuous)
    ncol = 4
    nrow = int(np.ceil(len(ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.6 * nrow), facecolor="white")
    for ax, identifier in zip(axes.flat, ids):
        column = continuous[identifier]
        g, r = grass[column], reference[column]
        gv, gw = g[g.notna()].to_numpy(), grass.loc[g.notna(), "area_weight_m2"].to_numpy()
        rv, rw = r[r.notna()].to_numpy(), reference.loc[r.notna(), "area_weight_m2"].to_numpy()
        bins = np.linspace(*wquantile(gv, gw, [0.005, 0.995]), 36)
        ax.hist(gv, bins=bins, weights=gw / gw.sum(), color="#d9d8d2", label="All natural grassland")
        ax.hist(rv, bins=bins, weights=rw / rw.sum(), histtype="step", color=BLUE, lw=2,
                label="Reference pixels")
        row = table.set_index("band").loc[identifier]
        ax.set_title(f"{row['variable'][:38]}\n{100 * row['grassland_inside_reference_5_95']:.0f}% of grassland "
                     f"within reference 5-95%", fontsize=8.5, loc="left")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, length=0)
    for ax in list(axes.flat)[len(ids):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"{config.display_name}: environmental range of reference pixels vs all natural grassland",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"{args.run}_variable_distributions.png", dpi=150)

    # Figure 2: joint environmental space (weighted PCA of all grassland).
    mean, cov = wmean_cov(G, weight_all)
    vals, vecs = np.linalg.eigh(cov)
    vecs = vecs[:, ::-1]
    pcs_g, pcs_r = (G - mean) @ vecs[:, :2], (R - mean) @ vecs[:, :2]
    explained = vals[::-1][:2] / vals.sum()
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor="white")
    hb = ax.hexbin(pcs_g[:, 0], pcs_g[:, 1], C=weight_all, reduce_C_function=np.sum, gridsize=45,
                   cmap="Greys", mincnt=1, bins="log")
    inside = d_ref <= limit95
    ax.scatter(pcs_r[inside, 0], pcs_r[inside, 1], s=5, color=BLUE, alpha=0.35, label="Reference pixels")
    ax.set_xlabel(f"Environmental PC1 ({100 * explained[0]:.0f}% of variance)")
    ax.set_ylabel(f"Environmental PC2 ({100 * explained[1]:.0f}% of variance)")
    ax.set_title(f"{config.display_name}: grassland (grey) vs reference (blue)\n"
                 f"{100 * joint_inside:.0f}% of grassland area lies inside the reference envelope",
                 fontsize=10.5, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(out / f"{args.run}_environmental_space.png", dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
