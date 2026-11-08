"""Compare Adirondack reference and departure scores with the APA land classification.

Usage:
    python -m scripts.tools.adirondack_landclass_check [--run adirondacks_forestv2]

Rasterizes the APA state land classes (Wilderness, Primitive, Canoe, Wild Forest, Intensive Use, ...)
and private land classes (Resource Management, Rural Use, ...) from data/external/adirondacks/
onto the 500 m forest grid, then tabulates per class: forest-zone area, reference area and share,
and, when the run's integrity map exists, the mean departure percentile and the share beyond the
95th percentile. Writes outputs/adirondacks_landclass_check.csv.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
STATE = {7: "Wilderness", 8: "Canoe Area", 9: "Primitive", 10: "Wild Forest", 11: "Intensive Use",
         12: "Historic", 13: "State Administrative", 14: "Pending Classification"}
PRIVATE = {1: "Hamlet", 2: "Moderate Intensity", 3: "Low Intensity", 4: "Rural Use",
           5: "Resource Management", 6: "Industrial Use", 0: "Pending Private"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="adirondacks_forestv2")
    args = parser.parse_args()
    name = args.run.replace("_forestv2", "")
    percentile_path = (ROOT / "outputs" / "reference_condition_inference" / f"{args.run}_predictors_only_2018"
                       / f"{args.run}_predictors_only_2018_reference_departure_percentile.tif")
    score = None
    with rasterio.open(ROOT / "forest_inputs" / "forest_masks_v2" / f"{name}_forest_mask_2018.tif") as src:
        grid_source = src
        mask_crs = src.crs
        if percentile_path.exists():  # use the integrity map's grid; the masks are read onto it
            with rasterio.open(percentile_path) as pct:
                score, transform, shape, crs = pct.read(1, masked=True), pct.transform, pct.shape, pct.crs
        else:
            transform, shape, crs = src.transform, src.shape, src.crs
        window = from_bounds(*array_bounds(shape[0], shape[1], transform), transform=src.transform)
        forest = src.read(1, window=window.round_offsets().round_lengths(), boundless=True, fill_value=0) == 1
    with rasterio.open(ROOT / "forest_inputs" / "forest_reference_v2" / f"{name}_reference_500m.tif") as src:
        window = from_bounds(*array_bounds(shape[0], shape[1], transform), transform=src.transform)
        reference = src.read(1, window=window.round_offsets().round_lengths(), boundless=True, fill_value=0) == 1
    classes = np.full(shape, "Outside classified land", dtype=object)
    for layer, labels, owner in ((0, PRIVATE, "Private"), (1, STATE, "State")):
        frame = gpd.read_file(ROOT / "data" / "external" / "adirondacks" / f"adirondack_landclass_layer{layer}.geojson").to_crs(crs)
        for code, label in labels.items():
            shapes = [(g, 1) for g in frame.loc[frame["LCCode"] == code, "geometry"]]
            if shapes:
                inside = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
                classes[inside] = f"{owner}: {label}"
    rows = []
    for label in sorted(set(classes[forest].tolist())):
        in_class = (classes == label) & forest
        row = {"class": label, "forest_km2": in_class.sum() * 0.25, "reference_km2": (in_class & reference).sum() * 0.25}
        row["reference_share_pct"] = 100 * row["reference_km2"] / row["forest_km2"] if row["forest_km2"] else np.nan
        if score is not None and score.shape == shape:
            values = score[in_class].compressed()
            row["scored_km2"] = values.size * 0.25
            row["mean_departure"] = float(values.mean()) if values.size else np.nan
            row["pct_beyond_95th"] = 100 * float((values >= 0.95).mean()) if values.size else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("forest_km2", ascending=False).round(2)
    out = ROOT / "outputs" / "adirondacks_landclass_check.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print("integrity map:", "found" if score is not None else "not yet (reference columns only)")


if __name__ == "__main__":
    main()
