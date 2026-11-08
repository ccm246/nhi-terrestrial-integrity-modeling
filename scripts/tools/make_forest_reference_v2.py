"""Forest reference sites from canopy cover, human modification and human influence.

Usage:
    python -m scripts.tools.make_forest_reference_v2 NAME [NAME ...] [--min-pixels 1000]

Needs data/habitat/<NAME>_habitat_2018.tif (scripts.tools.classify_habitat). A 500 m cell is
forest when the habitat raster says forest (codes 1 and 2: canopy > 20 % on >= half of its 30 m
pixels, Hansen tree cover 2000 minus loss through 2018). It is a reference site when, in
addition, it passes the grassland reference's two human-pressure tests at the strictest
(human modification, human influence) pair in LADDER that leaves at least ``--min-pixels``
reference cells (the strictest pair is kept if none does): mean human modification index <= the
first value, and annual HII / 7000 < the second in every reference year without two consecutive
failing years (the persistence rule of scripts.fetch_gee_raster_tiles).

Writes, on the same grid as the habitat raster:
  forest_inputs/forest_reference_v2/<NAME>_reference_500m.tif   1 = reference site
  forest_inputs/forest_masks_v2/<NAME>_forest_mask_2018.tif     1 = forest zone (inference mask)
The reference raster replaces band d01 in scripts.tools.fetch_forest_tiles.
"""
import argparse
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import numpy as np
import rasterio
from pyproj import CRS

from ..analysis_config import load_analysis_configuration
from ..raster_cache_utils import load_wgs84_aoi, select_intersecting_tiles

ROOT = Path(__file__).resolve().parents[2]
WORKERS, TIMEOUT_SECONDS, MAX_ATTEMPTS = 4, 300, 3
# (human modification max, human influence max): the ladder the grassland reference sweep uses.
LADDER = [(0.1, 0.08), (0.2, 0.15), (0.3, 0.2), (0.4, 0.25), (0.4, 0.3), (0.5, 0.35), (0.6, 0.4)]


def human_influence_ok(config, threshold: float) -> ee.Image:
    """1 where annual HII / 7000 < threshold, allowing no two consecutive failing reference years."""
    start, end = config.reference_start_year, config.reference_end_year
    collection = ee.ImageCollection(config.datasets["human_influence"])
    annual = [collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mean().divide(7000).lt(threshold)
              for year in range(start, end + 1)]
    pairs = [annual[i].Or(annual[i + 1]) for i in range(len(annual) - 1)]
    return ee.ImageCollection.fromImages(pairs).reduce(ee.Reducer.min()).eq(1)


def fetch_human_pressure(config, habitat):
    """Mean human modification and the HII test per ladder rung, on the habitat raster's grid."""
    grid = config.grid
    wkt = CRS.from_string(grid.crs).to_wkt(version="WKT1_GDAL")
    _, tiles = select_intersecting_tiles(load_wgs84_aoi(config.aoi_path), grid)
    modification = ee.Image(config.datasets["human_modification"]).rename("hm").toFloat()
    modification = modification.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(
        crs=wkt, scale=grid.pixel_size_meters)
    bands = [modification] + [human_influence_ok(config, hii).rename(f"hii{k}").toFloat()
                              for k, (_, hii) in enumerate(LADDER)]
    stack = ee.Image.cat(bands)
    names = ["hm"] + [f"hii{k}" for k in range(len(LADDER))]
    size, pixel = grid.tile_size_pixels, grid.pixel_size_meters

    def fetch(tile):
        request = {"expression": stack, "fileFormat": "NPY", "grid": {
            "dimensions": {"width": size, "height": size},
            "affineTransform": {"scaleX": pixel, "shearX": 0, "translateX": tile.left,
                                "shearY": 0, "scaleY": -pixel, "translateY": tile.top},
            "crsWkt": wkt}}
        data = np.load(io.BytesIO(ee.data.computePixels(request)), allow_pickle=False)
        return np.stack([data[n] for n in names])

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
    pool.shutdown(wait=False, cancel_futures=True)
    left, top = habitat.transform.c, habitat.transform.f
    out = np.full((len(names), habitat.height, habitat.width), np.nan, dtype="float32")
    for index, block in results.items():
        t = tiles[index]
        col0, row0 = round((t.left - left) / pixel), round((top - t.top) / pixel)
        out[:, row0:row0 + size, col0:col0 + size] = block
    return out[0], out[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--min-pixels", type=int, default=1000)
    parser.add_argument("--ee-project", default="charged-sled-403915")
    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)
    for name in args.names:
        config = load_analysis_configuration(ROOT / "config" / f"{name}_all.toml")
        with rasterio.open(ROOT / "data" / "habitat" / f"{name}_habitat_2018.tif") as src:
            habitat, profile = src.read(1), src.profile
            hm, hii = fetch_human_pressure(config, src)
        forest = (habitat == 1) | (habitat == 2)
        rungs = [forest & (hm <= modification) & (hii[k] == 1) for k, (modification, _) in enumerate(LADDER)]
        counts = [int(r.sum()) for r in rungs]
        chosen = next((k for k, c in enumerate(counts) if c >= args.min_pixels), None)
        used = chosen if chosen is not None else 0
        reference = rungs[used].astype("uint8")
        profile.update(count=1, dtype="uint8", compress="zstd", nodata=None)
        for array, folder, stem in ((reference, "forest_reference_v2", f"{name}_reference_500m"),
                                    (forest.astype("uint8"), "forest_masks_v2", f"{name}_forest_mask_2018")):
            path = ROOT / "forest_inputs" / folder / f"{stem}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", **profile) as sink:
                sink.write(array, 1)
        print(json.dumps({"name": name, "forest_km2": forest.sum() * 0.25,
                          "reference_km2_by_rung": {f"hm<={m},hii<{h}": c * 0.25 for (m, h), c in zip(LADDER, counts)},
                          "rung_used": {"human_modification": LADDER[used][0], "human_influence": LADDER[used][1]},
                          "reached_min_pixels": chosen is not None,
                          "reference_km2": float(reference.sum() * 0.25)}), flush=True)


if __name__ == "__main__":
    main()
