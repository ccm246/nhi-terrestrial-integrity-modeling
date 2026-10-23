"""One-page cross-landscape comparison (figure + CSV + Markdown table).

Usage:
    python -m scripts.tools.summarize_landscapes OUTPUT_DIR [RUN ...]

Reads, for each run, outputs/reference_coverage/<run>_summary.csv,
outputs/integrity_parameters/<run>_{predictors_only,all_variables}/response_metrics.csv
and the departure-percentile raster, and writes landscape_summary.{csv,md,png}.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "kaza": "KAZA", "pacha": "PACHA", "central_india_boundary": "Central India",
    "nepal_tal": "Nepal TAL", "soknot": "SOKNOT", "carpathian": "Carpathian", "sabah": "Sabah",
}
# Fixed categorical slots (reference palette order) so a landscape keeps its color in every panel.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#ecebe6"


def collect(run: str) -> dict:
    cov = pd.read_csv(ROOT / "outputs" / "reference_coverage" / f"{run}_summary.csv",
                      index_col=0)["value"]
    pred = pd.read_csv(ROOT / "outputs" / "integrity_parameters" / f"{run}_predictors_only"
                       / "response_metrics.csv")
    allv = pd.read_csv(ROOT / "outputs" / "integrity_parameters" / f"{run}_all_variables"
                       / "response_metrics.csv")
    folder = ROOT / "outputs" / "reference_condition_inference" / f"{run}_predictors_only_2018"
    with rasterio.open(next(folder.glob("*reference_departure_percentile.tif"))) as src:
        values = src.read(1, masked=True).compressed()
    return {
        "grassland_km2": cov["grassland_km2"], "reference_km2": cov["reference_km2"],
        "reference_share_pct": 100 * cov["reference_share_of_grassland"],
        "joint_envelope_pct": 100 * cov["grassland_inside_joint_95_envelope"],
        "median_variable_inside_pct": 100 * cov["median_var_inside_5_95"],
        "responses_fitted": len(pred),
        "mean_r2_predictors_only": pred["overall_weighted_r2"].mean(),
        "responses_r2_above_0.3": int((pred["overall_weighted_r2"] > 0.3).sum()),
        "mean_r2_all_variables": allv["overall_weighted_r2"].mean(),
        "scored_km2": len(values) * 0.25,
        "mean_departure": float(values.mean()),
        "pct_beyond_95th": 100 * float((values >= 0.95).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()
    runs = args.runs or list(RUNS)
    table = pd.DataFrame({RUNS.get(r, r): collect(r) for r in runs}).T
    table.index.name = "landscape"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(args.output_dir / "landscape_summary.csv")
    pretty = table.round(2).astype(str)
    lines = ["| landscape | " + " | ".join(pretty.columns) + " |", "|" + "---|" * (len(pretty.columns) + 1)]
    lines += [f"| {name} | " + " | ".join(row) + " |" for name, row in pretty.iterrows()]
    (args.output_dir / "landscape_summary.md").write_text("\n".join(lines) + "\n")

    order = list(table.sort_values("mean_r2_predictors_only", ascending=False).index)
    color = {name: COLORS[i % len(COLORS)] for i, name in enumerate(table.index)}
    panels = [
        ("reference_km2", "Reference area (km², log scale)", dict(log=True)),
        ("reference_share_pct", "Reference share of all grassland (%)", {}),
        ("joint_envelope_pct", "Grassland inside the reference envelope (%)\n(dashed line = 95%, a perfectly representative reference)", dict(ref=95, lo=40, hi=104)),
        ("mean_r2_predictors_only", "Held-out R²: environmental predictors only (bars)\nvs all variables (circles; inflated, responses predict each other)", dict(zero=True, extra="mean_r2_all_variables")),
        ("mean_departure", "Mean departure percentile (0 = like reference, 1 = far)", dict(lo=0, hi=1)),
        ("pct_beyond_95th", "Scored grassland beyond the 95th percentile (%)", {}),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), facecolor="white")
    y = np.arange(len(order))[::-1]
    for ax, (column, title, opt) in zip(axes.flat, panels):
        values = table.loc[order, column].to_numpy(float)
        ax.barh(y, values, color=[color[n] for n in order], height=0.62)
        if "extra" in opt:
            extra = table.loc[order, opt["extra"]].to_numpy(float)
            ax.scatter(extra, y, s=42, facecolor="white", edgecolor=MUTED, linewidth=1.4, zorder=3)
        if opt.get("log"):
            ax.set_xscale("log")
        if "ref" in opt:
            ax.axvline(opt["ref"], color=MUTED, lw=1, ls=(0, (4, 3)))
        if opt.get("zero"):
            ax.axvline(0, color=MUTED, lw=0.8)
        if "lo" in opt or "hi" in opt:
            ax.set_xlim(opt.get("lo"), opt.get("hi"))
        for yi, v in zip(y, values):
            label = f"{v:,.0f}" if column == "reference_km2" else (f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}")
            ax.text(v if v >= 0 else 0, yi, f" {label}", va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_yticks(y, order if ax in axes[:, 0] else [""] * len(order), fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, labelsize=8.5, colors=MUTED)
        ax.margins(x=0.18)
    fig.suptitle("Reference-condition models across landscapes (2018, grassland)", x=0.06, y=0.995,
                 ha="left", fontsize=14, color=INK)
    fig.text(0.06, 0.02, "Ordered by held-out R² of the environmental-predictor models. Departure statistics cover "
             "grassland pixels that are not themselves reference pixels. Low reference area and low R² mean "
             "the departure scores are weak evidence of integrity (Carpathian, Sabah).",
             fontsize=8.5, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(args.output_dir / "landscape_summary.png", dpi=160, facecolor="white")
    print(table.round(2).T.to_string())
    print("wrote", args.output_dir)


if __name__ == "__main__":
    main()
