"""Upper Midwest forest and grassland references from Wisconsin State Natural Areas.

Usage:
    python -m scripts.tools.make_upper_midwest_references [--shapefile PATH] [--hm-max 0.2]

Needs data/habitat/upper_midwest_habitat_2018.tif (scripts.tools.classify_habitat). The natural-area
polygons are rasterised onto the 500 m habitat grid (a cell counts when its centre falls inside a
polygon) and cleaned at the cell level, because the polygons also hold roads, fields, water and
recently cleared ground:
  * only cells the habitat classifier scores (forest 1/2 or grassland 3) are kept, which already drops
    cropland, built-up, water and Hansen canopy-loss cells (those are codes 0 and 4);
  * mean human modification must be <= --hm-max (default 0.2); human influence is not used.
The cleaned cells are split by habitat into a forest reference and a grassland reference. The zones
(inference masks) are the habitat classes over the whole four-state area, so Wisconsin trains the models
and they are applied across WI, MN, IA and IL.

Writes (habitat grid):
  forest_inputs/forest_reference_v2/upper_midwest_reference_500m.tif, forest_masks_v2/..._forest_mask_2018.tif
  grass_inputs/grass_reference_v2/upper_midwest_reference_500m.tif,   grass_masks_v2/..._grass_mask_2018.tif
"""
import argparse
import json
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from ..analysis_config import load_analysis_configuration
from .make_forest_reference_v2 import fetch_human_pressure

ROOT = Path(__file__).resolve().parents[2]
NAME = "upper_midwest"
DEFAULT_SHAPEFILE = Path.home() / "Downloads/Reference Polygons/wisconsinnaturalareas_references/SNA_BND_PUB.shp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapefile", default=str(DEFAULT_SHAPEFILE))
    parser.add_argument("--hm-max", type=float, default=0.2)
    parser.add_argument("--ee-project", default="charged-sled-403915")
    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)
    config = load_analysis_configuration(ROOT / "config" / f"{NAME}_all.toml")
    with rasterio.open(ROOT / "data" / "habitat" / f"{NAME}_habitat_2018.tif") as src:
        habitat, profile = src.read(1), src.profile
        hm, _ = fetch_human_pressure(config, src)
        areas = gpd.read_file(args.shapefile).to_crs(src.crs)
        inside = geometry_mask(areas.geometry, out_shape=habitat.shape, transform=src.transform, invert=True)
    forest, grass = np.isin(habitat, (1, 2)), habitat == 3
    clean = inside & (hm <= args.hm_max)
    stats = {"natural_area_cells_km2": float(inside.sum() * 0.25),
             "inside_and_scored_km2": float((inside & (forest | grass)).sum() * 0.25),
             "forest_reference_km2": float((clean & forest).sum() * 0.25),
             "grass_reference_km2": float((clean & grass).sum() * 0.25),
             "dropped_unscored_or_disturbed_km2": float((inside & ~(forest | grass)).sum() * 0.25),
             "dropped_high_modification_km2": float((inside & (forest | grass) & ~(hm <= args.hm_max)).sum() * 0.25)}
    print(json.dumps(stats, indent=1), flush=True)
    profile.update(count=1, dtype="uint8", compress="zstd", nodata=None)
    outputs = {("forest_inputs", "forest_reference_v2", f"{NAME}_reference_500m"): clean & forest,
               ("forest_inputs", "forest_masks_v2", f"{NAME}_forest_mask_2018"): forest,
               ("grass_inputs", "grass_reference_v2", f"{NAME}_reference_500m"): clean & grass,
               ("grass_inputs", "grass_masks_v2", f"{NAME}_grass_mask_2018"): grass}
    for (top, folder, stem), array in outputs.items():
        path = ROOT / top / folder / f"{stem}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as sink:
            sink.write(array.astype("uint8"), 1)


if __name__ == "__main__":
    main()
