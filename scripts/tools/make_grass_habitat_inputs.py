"""Grassland zone and reference sites from the forest/grassland habitat split.

Usage:
    python -m scripts.tools.make_grass_habitat_inputs NAME [NAME ...] [--dry-woodland] [--min-pixels 1000]

Needs data/habitat/<NAME>_habitat_2018.tif (scripts.tools.classify_habitat). A 500 m cell is in the
grassland zone when the habitat raster says grassland (code 3: natural, canopy <= 20 %). With
``--dry-woodland`` the cells coded 4 (ESA tree cover or mangrove whose Hansen canopy fails the 20 %
test: open dry woodland in savanna landscapes) are scored as grassland too. Do not use that flag where
code 4 is plantation or regrowing forest (Sabah, SEKA).

A grassland cell is a reference site when it also passes the human-pressure ladder used for the
forest reference (scripts.tools.make_forest_reference_v2.LADDER): the strictest (human modification,
human influence) pair that leaves at least ``--min-pixels`` reference cells.

Writes, on the habitat raster's grid:
  grass_inputs/grass_reference_v2/<NAME>_reference_500m.tif   1 = reference site
  grass_inputs/grass_masks_v2/<NAME>_grass_mask_2018.tif      1 = grassland zone (inference mask)
The reference raster replaces band d01 in scripts.tools.fetch_forest_tiles (pass --reference).
"""
import argparse
import json
from pathlib import Path

import ee
import rasterio

from ..analysis_config import load_analysis_configuration
from .make_forest_reference_v2 import LADDER, fetch_human_pressure

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--dry-woodland", action="store_true", help="Also score habitat code 4 as grassland.")
    parser.add_argument("--min-pixels", type=int, default=1000)
    parser.add_argument("--ee-project", default="charged-sled-403915")
    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)
    for name in args.names:
        config = load_analysis_configuration(ROOT / "config" / f"{name}_all.toml")
        with rasterio.open(ROOT / "data" / "habitat" / f"{name}_habitat_2018.tif") as src:
            habitat, profile = src.read(1), src.profile
            hm, hii = fetch_human_pressure(config, src)
        zone = (habitat == 3) | ((habitat == 4) if args.dry_woodland else False)
        rungs = [zone & (hm <= modification) & (hii[k] == 1) for k, (modification, _) in enumerate(LADDER)]
        counts = [int(r.sum()) for r in rungs]
        chosen = next((k for k, c in enumerate(counts) if c >= args.min_pixels), None)
        used = chosen if chosen is not None else 0
        reference = rungs[used].astype("uint8")
        profile.update(count=1, dtype="uint8", compress="zstd", nodata=None)
        for array, folder, stem in ((reference, "grass_reference_v2", f"{name}_reference_500m"),
                                    (zone.astype("uint8"), "grass_masks_v2", f"{name}_grass_mask_2018")):
            path = ROOT / "grass_inputs" / folder / f"{stem}.tif"
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(path, "w", **profile) as sink:
                sink.write(array, 1)
        print(json.dumps({"name": name, "dry_woodland_as_grassland": args.dry_woodland,
                          "grass_zone_km2": zone.sum() * 0.25,
                          "reference_km2_by_rung": {f"hm<={m},hii<{h}": c * 0.25 for (m, h), c in zip(LADDER, counts)},
                          "rung_used": {"human_modification": LADDER[used][0], "human_influence": LADDER[used][1]},
                          "reached_min_pixels": chosen is not None,
                          "reference_km2": float(reference.sum() * 0.25)}), flush=True)


if __name__ == "__main__":
    main()
