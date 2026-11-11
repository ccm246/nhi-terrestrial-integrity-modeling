"""Reference rasters and forest mask for the Adirondack transfer landscape (adirondacks_large).

Usage:
    python -m scripts.tools.make_transfer_inputs adirondacks_large [--wild-classes 7 8 9]

Needs data/habitat/<NAME>_habitat_2018.tif (scripts.tools.classify_habitat on the large extent), the
Adirondack forest v2 reference raster, and the Park Agency land classes in data/external/adirondacks.
Models are trained only on reference sites inside the Adirondack Park and then applied over the whole
forest zone of the larger extent (northwest Ontario and Quebec, the St. Lawrence and Champlain
lowlands, the Mohawk valley).

Writes transfer_inputs/<NAME>/:
  reference_hm_500m.tif    forest v2 reference (canopy > 20 %, human modification <= 0.1, human influence
                           < 0.08), the existing Adirondack raster placed on the large grid
  reference_wild_500m.tif  forest zone inside the Park Agency state classes listed in --wild-classes
                           (7 Wilderness, 8 Canoe Area, 9 Primitive); no human-pressure test
  forest_mask_2018.tif     forest zone (habitat codes 1 and 2) over the whole extent: where models are applied
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--wild-classes", type=int, nargs="+", default=[7, 8, 9])
    args = parser.parse_args()
    with rasterio.open(ROOT / "data" / "habitat" / f"{args.name}_habitat_2018.tif") as src:
        habitat, profile, transform, crs = src.read(1), src.profile, src.transform, src.crs
    shape = habitat.shape
    forest = (habitat == 1) | (habitat == 2)
    with rasterio.open(ROOT / "forest_inputs" / "forest_reference_v2" / "adirondacks_reference_500m.tif") as src:
        window = from_bounds(*array_bounds(shape[0], shape[1], transform), transform=src.transform)
        hm = src.read(1, window=window.round_offsets().round_lengths(), boundless=True, fill_value=0) == 1
    hm &= forest
    classes = gpd.read_file(ROOT / "data" / "external" / "adirondacks" / "adirondack_landclass_layer1.geojson").to_crs(crs)
    shapes = [(g, 1) for g in classes.loc[classes["LCCode"].isin(args.wild_classes), "geometry"]]
    wild = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool) & forest
    out = ROOT / "transfer_inputs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", compress="zstd", nodata=None)
    for array, stem in ((hm, "reference_hm_500m"), (wild, "reference_wild_500m"), (forest, "forest_mask_2018")):
        with rasterio.open(out / f"{stem}.tif", "w", **profile) as sink:
            sink.write(array.astype("uint8"), 1)
    print(json.dumps({"name": args.name, "forest_km2": forest.sum() * 0.25, "reference_hm_km2": hm.sum() * 0.25,
                      "reference_wild_km2": wild.sum() * 0.25, "wild_classes": args.wild_classes,
                      "wild_and_hm_km2": (wild & hm).sum() * 0.25}), flush=True)


if __name__ == "__main__":
    main()
