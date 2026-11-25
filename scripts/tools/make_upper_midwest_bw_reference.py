"""Upper Midwest forest reference variant: Wisconsin natural areas plus the Boundary Waters wilderness.

Usage:
    python -m scripts.tools.make_upper_midwest_bw_reference [--hm-max 0.2]

Adds forest cells inside the Boundary Waters Canoe Area Wilderness (WDPA, IUCN Ib, northern Minnesota
boreal forest; config/aoi_sources/boundary_waters_wilderness.geojson) to the Wisconsin State Natural
Areas forest reference (scripts.tools.make_upper_midwest_references), with the same cell-level cleaning:
forest habitat (codes 1, 2) and mean human modification <= --hm-max.

Writes forest_inputs/forest_reference_v2/upper_midwest_bw_reference_500m.tif, plus
config/upper_midwest_bw_forestv2_{all,predictors_only}.toml (same forest zone, own tile cache) and
config/aoi/upper_midwest_bw.geojson for the map scripts.
"""
import argparse
import re
import shutil
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from ..analysis_config import load_analysis_configuration
from .make_forest_reference_v2 import fetch_human_pressure

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hm-max", type=float, default=0.2)
    parser.add_argument("--ee-project", default="charged-sled-403915")
    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)
    config = load_analysis_configuration(ROOT / "config" / "upper_midwest_all.toml")
    with rasterio.open(ROOT / "data" / "habitat" / "upper_midwest_habitat_2018.tif") as src:
        habitat, profile = src.read(1), src.profile
        hm, _ = fetch_human_pressure(config, src)
        wilderness = gpd.read_file(ROOT / "config" / "aoi_sources" / "boundary_waters_wilderness.geojson").to_crs(src.crs)
        inside = geometry_mask(wilderness.geometry, out_shape=habitat.shape, transform=src.transform, invert=True)
    forest = np.isin(habitat, (1, 2))
    added = inside & forest & (hm <= args.hm_max)
    with rasterio.open(ROOT / "forest_inputs" / "forest_reference_v2" / "upper_midwest_reference_500m.tif") as src:
        wisconsin = src.read(1) == 1
    reference = wisconsin | added
    print(f"Wisconsin {wisconsin.sum() * 0.25:,.0f} km2 + Boundary Waters {added.sum() * 0.25:,.0f} km2 "
          f"(of {inside.sum() * 0.25:,.0f} km2 inside; forest {(inside & forest).sum() * 0.25:,.0f}) = {reference.sum() * 0.25:,.0f} km2", flush=True)
    profile.update(count=1, dtype="uint8", compress="zstd", nodata=None)
    with rasterio.open(ROOT / "forest_inputs" / "forest_reference_v2" / "upper_midwest_bw_reference_500m.tif", "w", **profile) as sink:
        sink.write(reference.astype("uint8"), 1)
    shutil.copy(ROOT / "config" / "aoi" / "upper_midwest.geojson", ROOT / "config" / "aoi" / "upper_midwest_bw.geojson")
    for variant in ("all", "predictors_only"):
        text = (ROOT / "config" / f"upper_midwest_forestv2_{variant}.toml").read_text()
        subs = [(rf'(?m)^name = "upper_midwest_forestv2_{variant}_2018"', f'name = "upper_midwest_bw_forestv2_{variant}_2018"'),
                (r'(?m)^display_name = "([^"\n]*?) forest v2 \(([^)]*)\)"', r'display_name = "\1 forest v2, Wisconsin + Boundary Waters reference (\2)"'),
                (r'(?m)^cache_directory = ".*"', 'cache_directory = "../data/gee_cache_upper_midwest_bw_forestv2"')]
        for pattern, replacement in subs:
            text, count = re.subn(pattern, replacement, text, count=1)
            if count != 1:
                raise ValueError(f"{variant}: pattern not found: {pattern}")
        (ROOT / "config" / f"upper_midwest_bw_forestv2_{variant}.toml").write_text(text)


if __name__ == "__main__":
    main()
