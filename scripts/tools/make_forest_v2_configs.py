"""Forest configs for the canopy + human-pressure reference (forest v2).

Usage:
    python -m scripts.tools.make_forest_v2_configs NAME [NAME ...]

Writes config/<NAME>_forestv2_{all,predictors_only}.toml from config/<NAME>_{all,predictors_only}.toml:
own analysis name and tile cache (data/gee_cache_<NAME>_forestv2), the forest zone from
scripts.tools.make_forest_reference_v2 as the inference mask (forest_inputs/forest_masks_v2), and
only the responses that are not grassland-specific or part of the reference definition:
short vegetation height (d08), tree cover (d09) and non-tree vegetation cover (d10) are dropped.
The reference raster itself replaces band d01 in scripts.tools.fetch_forest_tiles (pass --reference).
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESPONSES = ["d02", "d03", "d04", "d05", "d06", "d07", "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19"]
# (validation block size in metres, fold count): 5 folds need that many reference-containing blocks.
# Blocks holding reference at 50 km: Nepal TAL 2026 13, Adirondacks 17, SEKA 15; Sabah 14 at 100 km.
CV = {"nepal_tal_2026": (50000, 5), "adirondacks": (50000, 5), "seka": (50000, 5)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    responses = "responses = [" + ", ".join(f'"{r}"' for r in RESPONSES) + "]"
    for name in args.names:
        for variant in ("all", "predictors_only"):
            text = (ROOT / "config" / f"{name}_{variant}.toml").read_text()
            subs = [
                (rf'(?m)^name = "{name}_{variant}_2018"', f'name = "{name}_forestv2_{variant}_2018"'),
                (r'(?m)^display_name = "([^"\n]*?) \(([^)]*)\)"', r'display_name = "\1 forest v2 (\2)"'),
                (r'(?m)^cache_directory = ".*"', f'cache_directory = "../data/gee_cache_{name}_forestv2"'),
                (r'(?m)^application_mask_path = ".*"', f'application_mask_path = "../forest_inputs/forest_masks_v2/{name}_forest_mask_2018.tif"'),
                (r'(?m)^application_mask_label = ".*"', 'application_mask_label = "2018 forest zone (canopy > 20%)"'),
                (r'(?m)^display_name = "Grassland reference sites"', 'display_name = "Forest reference sites (canopy + human pressure)"'),
                (r'(?m)^responses = \[.*\]', responses),
            ]
            if name in CV:
                block, folds = CV[name]
                subs += [(r'(?m)^validation_block_size_meters = \d+', f'validation_block_size_meters = {block}'),
                         (r'(?m)^fold_count = \d+', f'fold_count = {folds}')]
            for pattern, replacement in subs:
                text, count = re.subn(pattern, replacement, text, count=1)
                if count != 1:
                    raise ValueError(f"{name}_{variant}: pattern not found: {pattern}")
            (ROOT / "config" / f"{name}_forestv2_{variant}.toml").write_text(text)
        print("wrote", f"config/{name}_forestv2_{{all,predictors_only}}.toml")


if __name__ == "__main__":
    main()
