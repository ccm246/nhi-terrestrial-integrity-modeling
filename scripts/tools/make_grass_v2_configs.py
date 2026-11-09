"""Grassland configs for the habitat-split grassland zone and reference (grass v2).

Usage:
    python -m scripts.tools.make_grass_v2_configs NAME [NAME ...]

Writes config/<NAME>_grassv2_{all,predictors_only}.toml from config/<NAME>_{all,predictors_only}.toml:
own analysis name and tile cache (data/gee_cache_<NAME>_grassv2) and the grassland zone from
scripts.tools.make_grass_habitat_inputs as the inference mask (grass_inputs/grass_masks_v2).
Responses and cross-validation settings are unchanged. The reference raster itself replaces band
d01 in scripts.tools.fetch_forest_tiles (pass --reference).
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        for variant in ("all", "predictors_only"):
            text = (ROOT / "config" / f"{name}_{variant}.toml").read_text()
            subs = [
                (rf'(?m)^name = "{name}_{variant}_2018"', f'name = "{name}_grassv2_{variant}_2018"'),
                (r'(?m)^display_name = "([^"\n]*?) \(([^)]*)\)"', r'display_name = "\1 grassland v2 (\2)"'),
                (r'(?m)^cache_directory = ".*"', f'cache_directory = "../data/gee_cache_{name}_grassv2"'),
                (r'(?m)^application_mask_path = ".*"', f'application_mask_path = "../grass_inputs/grass_masks_v2/{name}_grass_mask_2018.tif"'),
                (r'(?m)^application_mask_label = ".*"', 'application_mask_label = "2018 grassland zone (habitat split)"'),
            ]
            for pattern, replacement in subs:
                text, count = re.subn(pattern, replacement, text, count=1)
                if count != 1:
                    raise ValueError(f"{name}_{variant}: pattern not found: {pattern}")
            (ROOT / "config" / f"{name}_grassv2_{variant}.toml").write_text(text)
        print("wrote", f"config/{name}_grassv2_{{all,predictors_only}}.toml")


if __name__ == "__main__":
    main()
