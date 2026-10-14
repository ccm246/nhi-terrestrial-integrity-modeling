"""Build a per-landscape grassland application mask on the cache grid.

Usage:
    python -m scripts.tools.make_grassland_mask CONFIG.toml \
        [--probability-threshold 30] [--output PATH]

Takes the Global Pasture Watch natural/semi-natural grassland probability layer
for the analysis year, marks 30 m pixels at or above the threshold, and takes the
majority (>= 50 %) at 500 m on the same EPSG:6933 grid as the cached tiles. The
result is a single 0/1 GeoTIFF covering every cached tile of the analysis, ready
for ``[inference].application_mask_path`` (value 1 = inference pixel).

The default threshold is the config's reference grassland probability, so the
mask is consistent with how that landscape's reference sites were defined.
"""
import argparse
import tempfile
from pathlib import Path

import ee
import rasterio
from pyproj import CRS
from rasterio.merge import merge

from ..analysis_config import load_analysis_configuration
from ..raster_cache_utils import resolve_analysis_cache_tiles


def build_mask_image(dataset_id: str, year: int, threshold: float) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    probability = ee.Image(
        ee.ImageCollection(dataset_id).filterDate(start, start.advance(1, "year")).first()
    ).select(0)
    return probability.gte(threshold).unmask(0).toFloat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_configuration", type=Path)
    parser.add_argument("--probability-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_analysis_configuration(args.analysis_configuration)
    threshold = (
        args.probability_threshold
        if args.probability_threshold is not None
        else config.reference.grassland_probability
    )
    ee.Initialize(project=config.earth_engine.project)
    grid = config.grid
    wkt = CRS.from_string(grid.crs).to_wkt(version="WKT1_GDAL")
    cache = resolve_analysis_cache_tiles(config, show_progress=False)
    source = build_mask_image(
        config.datasets["grassland_probability"], config.year, threshold
    )
    # Majority of 30 m pixels inside each 500 m cell.
    aggregated = source.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(
        crs=wkt, scale=grid.pixel_size_meters
    )
    mask = aggregated.gte(0.5).rename("mask").toByte()

    output = args.output or (
        config.path.parent.parent / "data" / "grassland_masks"
        / f"{config.path.stem}_grassland_mask_{config.year}.tif"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    size = grid.tile_size_pixels
    with tempfile.TemporaryDirectory() as workdir:
        tile_paths = []
        for index, cached in enumerate(cache.tiles, start=1):
            tile = cached.tile
            request = {
                "expression": mask,
                "fileFormat": "GEO_TIFF",
                "grid": {
                    "dimensions": {"width": size, "height": size},
                    "affineTransform": {
                        "scaleX": grid.pixel_size_meters, "shearX": 0,
                        "translateX": tile.left, "shearY": 0,
                        "scaleY": -grid.pixel_size_meters, "translateY": tile.top,
                    },
                    "crsWkt": wkt,
                },
            }
            path = Path(workdir) / f"{tile.tile_id}.tif"
            path.write_bytes(ee.data.computePixels(request))
            with rasterio.open(path) as check:
                count = int(check.read(1).sum())
            tile_paths.append(path)
            print(f"tile {index}/{len(cache.tiles)} {tile.tile_id}: "
                  f"{count} grassland px", flush=True)
        sources = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = merge(sources)
        profile = sources[0].profile | {
            "height": mosaic.shape[1], "width": mosaic.shape[2],
            "transform": transform, "compress": "zstd", "nodata": None,
        }
        for source_file in sources:
            source_file.close()
        with rasterio.open(output, "w", **profile) as sink:
            sink.write(mosaic)
    print(f"Wrote {output}  grassland pixels: {int(mosaic.sum()):,} "
          f"({mosaic.sum() * grid.pixel_size_meters ** 2 / 1e6:,.0f} km2)")


if __name__ == "__main__":
    main()
