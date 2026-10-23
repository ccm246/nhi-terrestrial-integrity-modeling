"""Forest configs: copies of each landscape's grassland configs for the forest workflow.

Usage:
    python -m scripts.tools.make_forest_configs NAME [NAME ...]

Writes config/<NAME>_forest_{all,predictors_only}.toml from config/<NAME>_{all,predictors_only}.toml
with a distinct analysis name, its own tile cache (data/gee_cache_<NAME>_forest), the forest
extent as the inference mask (forest_inputs/forest_masks/<NAME>_forest_mask_2018.tif), and the
reference band relabelled. The Earth-Engine reference thresholds left in the file are unused by
scripts.tools.fetch_forest_tiles (which replaces d01 with forest_inputs/forest_reference/...).
All responses are modelled (`responses = []`).
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# (validation block size in metres, fold count) where the 100 km / 5-fold default has too few
# reference-containing blocks (blocks holding sampled reference rows: SEKA 4 at 75 km, SOKNOT 2 at 100 km, Sabah 6 at 100 km).
CV_OVERRIDES = {"seka": (75000, 4), "soknot": (100000, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    for name in args.names:
        for variant in ("all", "predictors_only"):
            text = (ROOT / "config" / f"{name}_{variant}.toml").read_text()
            subs = [
                (rf'(?m)^name = "{name}_{variant}_2018"', f'name = "{name}_forest_{variant}_2018"'),
                (r'(?m)^display_name = "([^"\n]*?) \(([^)]*)\)"', r'display_name = "\1 forest (\2)"'),
                (r'(?m)^cache_directory = ".*"', f'cache_directory = "../data/gee_cache_{name}_forest"'),
                (r'(?m)^application_mask_path = ".*"', f'application_mask_path = "../forest_inputs/forest_masks/{name}_forest_mask_2018.tif"'),
                (r'(?m)^application_mask_label = ".*"', 'application_mask_label = "2018 forest extent (IFL template)"'),
                (r'(?m)^display_name = "Grassland reference sites"', 'display_name = "Forest reference sites (IFL + high integrity)"'),
                (r'(?m)^responses = \[.*\]', 'responses = []'),
            ]
            if name in CV_OVERRIDES:
                block, folds = CV_OVERRIDES[name]
                subs += [(r'(?m)^validation_block_size_meters = \d+', f'validation_block_size_meters = {block}'),
                         (r'(?m)^fold_count = \d+', f'fold_count = {folds}')]
            for pattern, replacement in subs:
                text, count = re.subn(pattern, replacement, text, count=1)
                if count != 1:
                    raise ValueError(f"{name}_{variant}: pattern not found: {pattern}")
            (ROOT / "config" / f"{name}_forest_{variant}.toml").write_text(text)
        print("wrote", f"config/{name}_forest_{{all,predictors_only}}.toml")


if __name__ == "__main__":
    main()
