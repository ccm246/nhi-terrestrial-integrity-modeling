"""Count reference-site pixels in an AOI for several reference-threshold sets.

Usage:
    python -m scripts.tools.reference_threshold_sweep CONFIG.toml

Builds the d01 reference layer in Earth Engine with each (grassland probability,
human modification, human influence) triple and counts 500 m pixels inside the
AOI, so a landscape's thresholds can be chosen from measured counts.
"""
import argparse
import dataclasses
import json
from pathlib import Path

import ee

from ..analysis_config import load_analysis_configuration
from ..fetch_gee_raster_tiles import build_earth_engine_stack

THRESHOLDS = [
    (80, 0.1, 0.08),
    (60, 0.2, 0.15),
    (50, 0.3, 0.2),
    (40, 0.4, 0.25),
    (30, 0.4, 0.3),
    (20, 0.5, 0.35),
    (10, 0.6, 0.4),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_configuration", type=Path)
    args = parser.parse_args()
    config = load_analysis_configuration(args.analysis_configuration)
    ee.Initialize(project=config.earth_engine.project)
    aoi = json.load(open(config.aoi_path))
    region = ee.Geometry(aoi["features"][0]["geometry"])
    print("grassland_p  HMI   HII   reference_pixels(500m)")
    for gp, hm, hii in THRESHOLDS:
        reference = dataclasses.replace(
            config.reference,
            grassland_probability=gp,
            human_modification=hm,
            human_influence=hii,
        )
        trial = dataclasses.replace(config, reference=reference)
        image = build_earth_engine_stack(trial)
        d01 = image.select(0).unmask(0)
        count = d01.reduceRegion(
            ee.Reducer.sum(), region, scale=500,
            maxPixels=1e9, tileScale=4,
        ).values().get(0).getInfo()
        print(f"{gp:>11}  {hm:<4}  {hii:<5} {count}", flush=True)


if __name__ == "__main__":
    main()
