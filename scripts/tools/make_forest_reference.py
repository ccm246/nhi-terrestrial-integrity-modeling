"""Build forest reference and forest-extent rasters on the pipeline's 500 m grid.

Usage:
    python -m scripts.tools.make_forest_reference NAME [NAME ...] \
        [--source "~/Downloads/Forest Reference Layer/Reference_class.tif"] [--min-fraction 0.5]

The source is the forest-team raster (EPSG:4326, ~300 m) with classes
0 = non-forest, 1 = forest, 2 = Intact Forest Landscape (IFL), 3 = IFL with high
integrity (FLII >= 9.6). For each landscape (config/aoi/<NAME>.geojson) every 500 m cell
of the EPSG:6933 cache grid is set by the share of 300 m source pixels it contains:
  forest_inputs/forest_reference/<NAME>_reference_500m.tif  1 = class 3 share >= min-fraction
  forest_inputs/forest_masks/<NAME>_forest_mask_2018.tif    1 = forest (class >= 1) share >= min-fraction
Both are uint8 GeoTIFFs on the same global-anchored grid as the Earth Engine cache tiles,
so they can replace the Earth Engine reference band (d01) and serve as the inference mask.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

from ..analysis_config import load_analysis_configuration
from ..raster_cache_utils import load_wgs84_aoi, select_intersecting_tiles

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "Downloads" / "Forest Reference Layer" / "Reference_class.tif"


def tile_fractions(source, tile, grid, to_lonlat):
    """Return (reference_share, forest_share) arrays for one cache tile."""
    size, pixel = grid.tile_size_pixels, grid.pixel_size_meters
    west, south, east, north = to_lonlat.transform_bounds(tile.left, tile.bottom, tile.right, tile.top, densify_pts=21)
    margin = 0.01
    window = from_bounds(west - margin, south - margin, east + margin, north + margin,
                         transform=source.transform).round_offsets().round_lengths()
    classes = source.read(1, window=window, boundless=True, fill_value=np.nan)
    classes = np.nan_to_num(classes, nan=0.0)
    src_transform = source.window_transform(window)
    dst_transform = from_origin(tile.left, tile.top, pixel, pixel)
    out = []
    for binary in ((classes == 3), (classes >= 1)):
        destination = np.zeros((size, size), dtype="float32")
        reproject(binary.astype("float32"), destination, src_transform=src_transform,
                  src_crs="EPSG:4326", dst_transform=dst_transform, dst_crs=grid.crs,
                  resampling=Resampling.average)
        out.append(destination)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--min-fraction", type=float, default=0.5)
    args = parser.parse_args()
    reference_dir = ROOT / "forest_inputs" / "forest_reference"
    mask_dir = ROOT / "forest_inputs" / "forest_masks"
    reference_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.source.expanduser()) as source:
        for name in args.names:
            config = load_analysis_configuration(ROOT / "config" / f"{name}_all.toml")
            grid = config.grid
            to_lonlat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
            projected_aoi, tiles = select_intersecting_tiles(load_wgs84_aoi(config.aoi_path), grid)
            left = min(t.left for t in tiles); right = max(t.right for t in tiles)
            top = max(t.top for t in tiles); bottom = min(t.bottom for t in tiles)
            pixel, size = grid.pixel_size_meters, grid.tile_size_pixels
            width, height = round((right - left) / pixel), round((top - bottom) / pixel)
            reference = np.zeros((height, width), dtype="uint8")
            forest = np.zeros((height, width), dtype="uint8")
            for tile in tiles:
                ref_share, forest_share = tile_fractions(source, tile, grid, to_lonlat)
                col0, row0 = round((tile.left - left) / pixel), round((top - tile.top) / pixel)
                reference[row0:row0 + size, col0:col0 + size] = ref_share >= args.min_fraction
                forest[row0:row0 + size, col0:col0 + size] = forest_share >= args.min_fraction
            profile = {"driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": "uint8",
                       "crs": CRS.from_string(grid.crs).to_wkt(), "compress": "zstd",
                       "transform": from_origin(left, top, pixel, pixel)}
            for array, path in ((reference, reference_dir / f"{name}_reference_500m.tif"),
                                (forest, mask_dir / f"{name}_forest_mask_2018.tif")):
                with rasterio.open(path, "w", **profile) as sink:
                    sink.write(array, 1)
            inside = geometry_mask([projected_aoi], out_shape=(height, width),
                                   transform=from_origin(left, top, pixel, pixel), invert=True)
            km2 = pixel * pixel / 1e6
            print(f"{name:10} {len(tiles):4d} tiles | inside AOI: reference {int((reference & inside).sum()) * km2:>9,.0f} km2"
                  f" | forest {int((forest & inside).sum()) * km2:>10,.0f} km2 of {int(inside.sum()) * km2:>10,.0f} km2", flush=True)


if __name__ == "__main__":
    main()
