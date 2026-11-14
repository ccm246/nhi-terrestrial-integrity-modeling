"""Split a landscape into forest and grassland habitat on the pipeline's 500 m grid.

Usage:
    python -m scripts.tools.classify_habitat NAME [NAME ...] [--canopy 20] [--min-fraction 0.5]

Decision rule per 500 m cell (config/<NAME>_all.toml supplies the grid and AOI):
  1. Not natural (ESA WorldCover cropland, built-up, bare, snow or water covers >= min-fraction
     of the cell): left unscored (0).
  2. Natural and canopy cover > ``--canopy`` % (Hansen tree cover 2000 minus loss through 2018)
     on >= min-fraction of the cell's 30 m pixels: forest. Otherwise: grassland.
Canopy cover decides the habitat, so savanna and woodland that ESA labels as shrub or grass
go to whichever habitat their canopy is nearest to. Cells ESA calls tree cover or mangrove
whose canopy fails the test (plantations, logged or regrowing forest) are neither habitat and
stay unscored (4). Cells where ESA and canopy disagree otherwise keep their own code.

Output data/habitat/<NAME>_habitat_2018.tif (uint8):
  0 unscored (not natural)   1 forest (ESA agrees)   2 forest by canopy (ESA says not tree/mangrove)
  3 grassland   4 disturbed or regrowing tree cover (ESA tree/mangrove, canopy fails): unscored
"""
import argparse
import io
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import geometry_mask
from rasterio.transform import from_origin

from ..analysis_config import load_analysis_configuration
from ..raster_cache_utils import load_wgs84_aoi, select_intersecting_tiles

ROOT = Path(__file__).resolve().parents[2]
WORKERS, TIMEOUT_SECONDS, MAX_ATTEMPTS = 4, 600, 8
ESA_TREE, ESA_NONNATURAL = (10, 95), (40, 50, 60, 70, 80)  # tree cover, mangrove | crop, built, bare, snow, water
BANDS = ["nonnatural", "esa_tree", "canopy"]


def build_image(canopy_percent: float) -> ee.Image:
    esa = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    loss_year = hansen.select("lossyear").unmask(0)  # masked wherever there was no loss
    canopy = hansen.select("treecover2000").gt(canopy_percent).And(loss_year.eq(0).Or(loss_year.gt(18)))
    return ee.Image.cat(esa.remap(list(ESA_NONNATURAL), [1] * len(ESA_NONNATURAL), 0).rename("nonnatural"),
                        esa.remap(list(ESA_TREE), [1, 1], 0).rename("esa_tree"),
                        canopy.rename("canopy")).toFloat()


def classify(name: str, canopy_percent: float, min_fraction: float) -> Path:
    config = load_analysis_configuration(ROOT / "config" / f"{name}_all.toml")
    grid = config.grid
    wkt = CRS.from_string(grid.crs).to_wkt(version="WKT1_GDAL")
    projected_aoi, tiles = select_intersecting_tiles(load_wgs84_aoi(config.aoi_path), grid)
    shares = build_image(canopy_percent).reduceResolution(ee.Reducer.mean(), maxPixels=16384).reproject(
        crs=wkt, scale=grid.pixel_size_meters)
    size, pixel = grid.tile_size_pixels, grid.pixel_size_meters

    def request_block(left, top, cells):
        request = {"expression": shares, "fileFormat": "NPY", "grid": {
            "dimensions": {"width": cells, "height": cells},
            "affineTransform": {"scaleX": pixel, "shearX": 0, "translateX": left,
                                "shearY": 0, "scaleY": -pixel, "translateY": top},
            "crsWkt": wkt}}
        data = np.load(io.BytesIO(ee.data.computePixels(request)), allow_pickle=False)
        return np.stack([data[b] for b in BANDS])

    checkpoint = ROOT / "data" / "habitat" / "tile_checkpoints" / f"{name}_{canopy_percent:g}"
    checkpoint.mkdir(parents=True, exist_ok=True)

    def fetch(tile):
        """One tile's three share bands; saved to disk the moment it arrives so a rerun resumes."""
        path = checkpoint / f"{tile.tile_id}.npy"
        if path.exists():
            return np.load(path)
        try:
            out = request_block(tile.left, tile.top, size)
        except ee.EEException as error:
            if "too large" not in str(error):  # high latitudes: too many 10 m pixels per request
                raise
            half = size // 2
            out = np.zeros((len(BANDS), size, size), dtype="float32")
            for row in (0, half):
                for col in (0, half):
                    out[:, row:row + half, col:col + half] = request_block(
                        tile.left + col * pixel, tile.top - row * pixel, half)
        np.save(path.with_suffix(".tmp.npy"), out)
        path.with_suffix(".tmp.npy").replace(path)
        return out

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    pending = {i: pool.submit(fetch, t) for i, t in enumerate(tiles)}
    attempts, results = {i: 1 for i in pending}, {}
    while pending:
        for index, future in list(pending.items()):
            try:
                results[index] = future.result(timeout=TIMEOUT_SECONDS)
                del pending[index]
            except Exception as error:
                if attempts[index] >= MAX_ATTEMPTS:
                    raise RuntimeError(f"tile {tiles[index].tile_id} failed: {error!r}") from error
                attempts[index] += 1
                pending[index] = pool.submit(fetch, tiles[index])
        print(f"  {name}: {len(results)}/{len(tiles)} tiles", flush=True)
    pool.shutdown(wait=False, cancel_futures=True)

    left = min(t.left for t in tiles); right = max(t.right for t in tiles)
    top = max(t.top for t in tiles); bottom = min(t.bottom for t in tiles)
    width, height = round((right - left) / pixel), round((top - bottom) / pixel)
    habitat = np.zeros((height, width), dtype="uint8")
    for index, block in results.items():
        t = tiles[index]
        nonnatural, esa_tree, canopy = block
        natural = nonnatural < min_fraction
        forest, esa_forest = canopy >= min_fraction, esa_tree >= min_fraction
        code = np.zeros((size, size), dtype="uint8")
        code[natural & forest & esa_forest] = 1
        code[natural & forest & ~esa_forest] = 2
        code[natural & ~forest & ~esa_forest] = 3
        code[natural & ~forest & esa_forest] = 4
        col0, row0 = round((t.left - left) / pixel), round((top - t.top) / pixel)
        habitat[row0:row0 + size, col0:col0 + size] = code
    transform = from_origin(left, top, pixel, pixel)
    inside = geometry_mask([projected_aoi], out_shape=(height, width), transform=transform, invert=True)
    habitat[~inside] = 0
    destination = ROOT / "data" / "habitat" / f"{name}_habitat_2018.tif"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", driver="GTiff", height=height, width=width, count=1, dtype="uint8",
                       crs=CRS.from_string(grid.crs).to_wkt(), transform=transform, compress="deflate", nodata=None) as sink:
        sink.write(habitat, 1)
    km2, total = pixel * pixel / 1e6, inside.sum() * pixel * pixel / 1e6
    counts = {c: int(((habitat == c) & inside).sum()) * km2 for c in range(5)}
    row = {"name": name, "aoi_km2": round(total), "unscored_km2": round(total - counts[1] - counts[2] - counts[3] - counts[4]),
           "disturbed_tree_km2": round(counts[4]),
           "forest_km2": round(counts[1] + counts[2]), "forest_by_canopy_km2": round(counts[2]),
           "grassland_km2": round(counts[3])}
    print(json.dumps(row), flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--canopy", type=float, default=20.0, help="Canopy cover (%%) a pixel must exceed to count as forest.")
    parser.add_argument("--min-fraction", type=float, default=0.5)
    parser.add_argument("--ee-project", default="charged-sled-403915")
    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)
    for name in args.names:
        classify(name, args.canopy, args.min_fraction)


if __name__ == "__main__":
    main()
