"""Copy a tile cache, swapping band d01 for a different reference raster (no Earth Engine needed).

Usage:
    python -m scripts.tools.clone_cache_with_reference TARGET_CONFIG.toml SOURCE_CACHE_DIR REFERENCE.tif

TARGET_CONFIG's cache_directory is filled with the tiles of SOURCE_CACHE_DIR whose band 1 is
replaced by REFERENCE.tif (1 = reference site), validated and recorded in the target manifest by the
same cache code scripts.tools.fetch_forest_tiles uses. Use it when two runs differ only in reference
sites, so the predictor bands are downloaded once.
"""
import argparse
from pathlib import Path

import rasterio

from ..analysis_config import load_analysis_configuration
from ..fetch_gee_raster_tiles import cache_aoi_tiles
from .fetch_forest_tiles import replace_reference_band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_config", type=Path)
    parser.add_argument("source_cache", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    config = load_analysis_configuration(args.target_config)

    def local_bytes(_stack, tile, _grid, _bands):
        matches = list(args.source_cache.glob(f"tiles/*/{tile.tile_id}.tif"))
        if len(matches) != 1:
            raise FileNotFoundError(f"{tile.tile_id}: {len(matches)} source tiles in {args.source_cache}")
        return matches[0].read_bytes()

    with rasterio.open(args.reference) as reference:
        cache_aoi_tiles(config, refresh=False, show_progress=False,
                        tile_fetcher=lambda stack, tile, grid, bands: replace_reference_band(
                            local_bytes(stack, tile, grid, bands), tile, reference))


if __name__ == "__main__":
    main()
