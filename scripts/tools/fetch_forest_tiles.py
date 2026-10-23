"""Fetch the Earth Engine stack for a forest landscape with a locally built reference band.

Usage:
    python -m scripts.tools.fetch_forest_tiles config/<NAME>_forest_all.toml \
        [--reference forest_inputs/forest_reference/<NAME>_reference_500m.tif] [--refresh]

Same cache, manifest and validation as ``scripts.fetch_gee_raster_tiles``; the only difference
is band d01. Instead of the grassland reference computed in Earth Engine, d01 is taken from the
local 500 m reference raster built by ``scripts.tools.make_forest_reference`` (1 = reference
site, otherwise masked). Replacing d01 with a constant in the Earth Engine request also drops
its (expensive) multi-year computation, so the request is cheaper than the grassland fetch.
"""
import argparse
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

from ..analysis_config import load_analysis_configuration
from ..fetch_gee_raster_tiles import build_earth_engine_stack, cache_aoi_tiles, fetch_tile_bytes

ROOT = Path(__file__).resolve().parents[2]


def replace_reference_band(tile_bytes: bytes, tile, reference) -> bytes:
    """Return the GeoTIFF with band 1 replaced by the local reference raster for this tile."""
    with MemoryFile(tile_bytes) as source_file, source_file.open() as source:
        profile, descriptions, tags = source.profile, source.descriptions, source.tags()
        nodata = source.nodata
        data = source.read(masked=True).astype("float32").filled(-np.inf if nodata is None else nodata)
    window = from_bounds(tile.left, tile.bottom, tile.right, tile.top, transform=reference.transform)
    local = reference.read(1, window=window.round_offsets().round_lengths(), boundless=True, fill_value=0)
    if local.shape != data.shape[1:]:
        raise ValueError(f"reference window {local.shape} != tile {data.shape[1:]} for {tile.tile_id}")
    fill = -np.inf if nodata is None else nodata
    data[0] = np.where(local == 1, 1.0, fill)
    with MemoryFile() as out_file:
        with out_file.open(**profile) as sink:
            sink.write(data)
            for index, description in enumerate(descriptions, start=1):
                sink.set_band_description(index, description)
            sink.update_tags(**tags)
        return out_file.read()


def make_tile_fetcher(stack, reference, bytes_fetcher=fetch_tile_bytes):
    def fetch(_ignored_stack, tile, cache_grid, band_names):
        return replace_reference_band(bytes_fetcher(stack, tile, cache_grid, band_names), tile, reference)
    return fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_configuration", type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    config = load_analysis_configuration(args.analysis_configuration)
    stem = args.analysis_configuration.stem
    name = stem.replace("_forest_predictors_only", "").replace("_forest_all", "").replace("_handoff", "")
    reference_path = args.reference or ROOT / "forest_inputs" / "forest_reference" / f"{name}_reference_500m.tif"
    ee.Initialize(project=config.earth_engine.project)
    band_names = config.band_names()
    stack = build_earth_engine_stack(config).addBands(
        ee.Image.constant(0).toFloat().rename(band_names[0]), overwrite=True
    )
    with rasterio.open(reference_path) as reference:
        cache_aoi_tiles(config, refresh=args.refresh, show_progress=not args.no_progress,
                        tile_fetcher=make_tile_fetcher(stack, reference))


if __name__ == "__main__":
    main()
