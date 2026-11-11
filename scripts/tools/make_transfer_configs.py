"""Configs for the Adirondack transfer runs: two reference types, one large forest zone.

Usage:
    python -m scripts.tools.make_transfer_configs NAME [--wild-block-km 50 --wild-folds 5] [--hm-block-km 50 --hm-folds 5]

Writes config/<NAME>_{hm,wild}_{all,predictors_only}.toml from config/<NAME>_{all,predictors_only}.toml
(scripts.tools.make_landscape_config): own analysis name and tile cache (data/gee_cache_<NAME>_<ref>),
the forest zone of the large extent from scripts.tools.make_transfer_inputs as the inference mask,
and the forest response list of make_forest_v2_configs. Reference rasters replace band d01 in
scripts.tools.fetch_forest_tiles (or scripts.tools.clone_cache_with_reference).
"""
import argparse
import re
from pathlib import Path

from .make_forest_v2_configs import RESPONSES

ROOT = Path(__file__).resolve().parents[2]
LABEL = {"hm": "Adirondack reference (canopy + human pressure)", "wild": "Adirondack wilderness reference"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--hm-block-km", type=int, default=50)
    parser.add_argument("--hm-folds", type=int, default=5)
    parser.add_argument("--wild-block-km", type=int, default=50)
    parser.add_argument("--wild-folds", type=int, default=5)
    args = parser.parse_args()
    name = args.name
    responses = "responses = [" + ", ".join(f'"{r}"' for r in RESPONSES) + "]"
    cv = {"hm": (args.hm_block_km, args.hm_folds), "wild": (args.wild_block_km, args.wild_folds)}
    for ref in ("hm", "wild"):
        for variant in ("all", "predictors_only"):
            text = (ROOT / "config" / f"{name}_{variant}.toml").read_text()
            subs = [
                (rf'(?m)^name = "{name}_{variant}_2018"', f'name = "{name}_{ref}_{variant}_2018"'),
                (r'(?m)^display_name = "([^"\n]*?) \(([^)]*)\)"', rf'display_name = "\1 {ref} reference (\2)"'),
                (r'(?m)^cache_directory = ".*"', f'cache_directory = "../data/gee_cache_{name}_{ref}"'),
                (r'(?m)^application_mask_path = ".*"', f'application_mask_path = "../transfer_inputs/{name}/forest_mask_2018.tif"'),
                (r'(?m)^application_mask_label = ".*"', 'application_mask_label = "2018 forest zone, large extent (canopy > 20%)"'),
                (r'(?m)^display_name = "Grassland reference sites"', f'display_name = "{LABEL[ref]}"'),
                (r'(?m)^responses = \[.*\]', responses),
                (r'(?m)^validation_block_size_meters = \d+', f'validation_block_size_meters = {cv[ref][0] * 1000}'),
                (r'(?m)^fold_count = \d+', f'fold_count = {cv[ref][1]}'),
            ]
            for pattern, replacement in subs:
                text, count = re.subn(pattern, replacement, text, count=1)
                if count != 1:
                    raise ValueError(f"{name}_{ref}_{variant}: pattern not found: {pattern}")
            (ROOT / "config" / f"{name}_{ref}_{variant}.toml").write_text(text)
        print("wrote", f"config/{name}_{ref}_{{all,predictors_only}}.toml")


if __name__ == "__main__":
    main()
