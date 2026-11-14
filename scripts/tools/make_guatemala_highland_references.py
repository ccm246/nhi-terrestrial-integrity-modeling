"""Guatemala forest reference variants that include highland forest.

Usage:
    python -m scripts.tools.make_guatemala_highland_references

The standard ladder (scripts.tools.make_forest_reference_v2) stops at human modification <= 0.1 /
influence < 0.08, which leaves only ~900 km2 of highland forest (highland small-holder settlement lifts
the index) against ~9,500 km2 in the Peten lowlands. Two variants add highland forest (latitude <= 16.5 N)
to the existing strict reference (forest_reference_v2/guatemala_reference_500m.tif):

  pa   highland forest inside strictly protected areas (WDPA IUCN Ib, II, III, IV; designated),
       passing human modification <= 0.3 and human influence < 0.2
       (config/aoi_sources/guatemala_strict_protected_areas.geojson)
  mid  all highland forest passing human modification <= 0.2 and human influence < 0.15

Writes forest_inputs/forest_reference_v2/guatemala_{pa,mid}_reference_500m.tif on the habitat grid, plus
config/guatemala_{pa,mid}_forestv2_{all,predictors_only}.toml (same forest-zone mask, own tile cache) and
config/aoi/guatemala_{pa,mid}.geojson for the map scripts.
"""
import re
import shutil
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask

from ..analysis_config import load_analysis_configuration
from .make_forest_reference_v2 import LADDER, fetch_human_pressure

ROOT = Path(__file__).resolve().parents[2]
HIGHLAND_MAX_LAT = 16.5
# (human modification max, ladder index for the human-influence test)
PA_RUNG, MID_RUNG = 2, 1


def main() -> None:
    ee.Initialize(project="charged-sled-403915")
    config = load_analysis_configuration(ROOT / "config" / "guatemala_all.toml")
    with rasterio.open(ROOT / "data" / "habitat" / "guatemala_habitat_2018.tif") as src:
        habitat, profile = src.read(1), src.profile
        hm, hii = fetch_human_pressure(config, src)
        height, width = habitat.shape
        rows, cols = np.mgrid[0:height, 0:width]
        _, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(
            src.transform.c + (cols + 0.5) * src.transform.a, src.transform.f + (rows + 0.5) * src.transform.e)
        protected = gpd.read_file(ROOT / "config" / "aoi_sources" / "guatemala_strict_protected_areas.geojson").to_crs(src.crs)
        in_pa = geometry_mask(protected.geometry, out_shape=habitat.shape, transform=src.transform, invert=True)
    forest = np.isin(habitat, (1, 2))
    highland = forest & (lat <= HIGHLAND_MAX_LAT)
    with rasterio.open(ROOT / "forest_inputs" / "forest_reference_v2" / "guatemala_reference_500m.tif") as src:
        strict = src.read(1) == 1
    assert strict.shape == habitat.shape
    variants = {
        "pa": strict | (highland & in_pa & (hm <= LADDER[PA_RUNG][0]) & (hii[PA_RUNG] == 1)),
        "mid": strict | (highland & (hm <= LADDER[MID_RUNG][0]) & (hii[MID_RUNG] == 1)),
    }
    profile.update(count=1, dtype="uint8", compress="zstd", nodata=None)
    for tag, reference in variants.items():
        path = ROOT / "forest_inputs" / "forest_reference_v2" / f"guatemala_{tag}_reference_500m.tif"
        with rasterio.open(path, "w", **profile) as sink:
            sink.write(reference.astype("uint8"), 1)
        print(f"{tag}: reference {reference.sum() * 0.25:,.0f} km2 (strict {strict.sum() * 0.25:,.0f}; highland share "
              f"{(reference & highland).sum() / reference.sum():.0%})", flush=True)
        shutil.copy(ROOT / "config" / "aoi" / "guatemala.geojson", ROOT / "config" / "aoi" / f"guatemala_{tag}.geojson")
        for variant in ("all", "predictors_only"):
            text = (ROOT / "config" / f"guatemala_forestv2_{variant}.toml").read_text()
            subs = [(rf'(?m)^name = "guatemala_forestv2_{variant}_2018"', f'name = "guatemala_{tag}_forestv2_{variant}_2018"'),
                    (r'(?m)^display_name = "([^"\n]*?) forest v2 \(([^)]*)\)"', rf'display_name = "\1 forest v2, {tag} reference (\2)"'),
                    (r'(?m)^cache_directory = ".*"', f'cache_directory = "../data/gee_cache_guatemala_{tag}_forestv2"')]
            for pattern, replacement in subs:
                text, count = re.subn(pattern, replacement, text, count=1)
                if count != 1:
                    raise ValueError(f"{tag}/{variant}: pattern not found: {pattern}")
            (ROOT / "config" / f"guatemala_{tag}_forestv2_{variant}.toml").write_text(text)


if __name__ == "__main__":
    main()
