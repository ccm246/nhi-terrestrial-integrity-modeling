"""Count reference-site pixels in an AOI for several reference-threshold sets.

Usage:
    python -m scripts.tools.reference_threshold_sweep CONFIG.toml [--min-pixels N]

Builds the d01 reference layer in Earth Engine with each (grassland probability,
human modification, human influence) triple in LADDER (strictest first) and
counts 500 m pixels inside the AOI. The last output line is
``CHOSEN <grassland_p> <hmi> <hii> <index>`` for the strictest triple with at
least ``--min-pixels`` reference pixels (``CHOSEN none`` if no triple qualifies).
"""
import argparse
import dataclasses
import json
from pathlib import Path

import ee

from ..analysis_config import load_analysis_configuration
from ..fetch_gee_raster_tiles import build_earth_engine_stack

LADDER = [
    (80, 0.1, 0.08),
    (60, 0.2, 0.15),
    (50, 0.3, 0.2),
    (40, 0.4, 0.25),
    (30, 0.4, 0.3),
    (20, 0.5, 0.35),
    (10, 0.6, 0.4),
]


def count_reference_pixels(config, region, triple, attempts=3):
    gp, hm, hii = triple
    reference = dataclasses.replace(
        config.reference,
        grassland_probability=gp,
        human_modification=hm,
        human_influence=hii,
    )
    image = build_earth_engine_stack(dataclasses.replace(config, reference=reference))
    d01 = image.select(0).unmask(0)
    for attempt in range(attempts):
        try:
            return d01.reduceRegion(
                ee.Reducer.sum(), region, scale=500,
                maxPixels=1e10, tileScale=8,
            ).values().get(0).getInfo()
        except Exception as error:
            print(f"  attempt {attempt + 1} failed: {error!r}", flush=True)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_configuration", type=Path)
    parser.add_argument("--min-pixels", type=float, default=3000)
    args = parser.parse_args()
    config = load_analysis_configuration(args.analysis_configuration)
    ee.Initialize(project=config.earth_engine.project)
    aoi = json.load(open(config.aoi_path))
    region = ee.Geometry(aoi["features"][0]["geometry"])
    print("grassland_p  HMI   HII   reference_pixels(500m)")
    chosen = None
    for index, triple in enumerate(LADDER):
        count = count_reference_pixels(config, region, triple)
        print(f"{triple[0]:>11}  {triple[1]:<4}  {triple[2]:<5} {count}", flush=True)
        if chosen is None and count is not None and count >= args.min_pixels:
            chosen = (*triple, index)
            break
    print("CHOSEN " + (" ".join(map(str, chosen)) if chosen else "none"))


if __name__ == "__main__":
    main()
