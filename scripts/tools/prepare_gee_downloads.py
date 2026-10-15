"""Download every Earth Engine input the landscape runs need, then zip it for hand-off.

Run this on any machine/Earth Engine project with quota; send the resulting zips back
and unzip them into the repository root (they contain ``data/...`` paths).

Usage (from the repository root, after ``earthengine authenticate``):
    python -m scripts.tools.prepare_gee_downloads --ee-project YOUR_EE_PROJECT \
        [--only sabah nepal_tal ...] [--dry-run]

For each landscape in LANDSCAPES (config/<name>_all.toml, thresholds already
fixed in the committed configs) it:
  1. fetches the 2018 d01-d39 stack as 500 m, 128x128-px EPSG:6933 GeoTIFF tiles
     (resumable; re-run after any interruption),
  2. builds the landscape's grassland mask GeoTIFF,
  3. writes ``handoff/<name>.zip`` containing both.
Landscapes are done smallest first, so partial progress is still useful.
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# name -> approximate 64 km tile count (smallest first).
LANDSCAPES = {
    "nepal_tal": 20, "sabah": 33, "soknot": 67, "carpathian": 90,
    "kaza": 179, "pacha": 453,
}
FETCH_ATTEMPTS = 6


def run(command: list[str]) -> bool:
    print("$", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode == 0


def project_config(name: str, project: str) -> Path:
    """Copy config/<name>_all.toml with the Earth Engine project replaced."""
    text = (ROOT / "config" / f"{name}_all.toml").read_text()
    lines = [
        f'project = "{project}"' if line.strip().startswith("project = ") else line
        for line in text.splitlines()
    ]
    path = ROOT / "config" / f"{name}_handoff.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ee-project", required=True)
    parser.add_argument("--only", nargs="*", choices=list(LANDSCAPES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = [n for n in LANDSCAPES if not args.only or n in args.only]
    (ROOT / "handoff").mkdir(exist_ok=True)
    failed = []
    for name in names:
        print(f"\n=== {name} (~{LANDSCAPES[name]} tiles) ===", flush=True)
        if args.dry_run:
            continue
        config = project_config(name, args.ee_project)
        mask = ROOT / "data" / "grassland_masks" / f"{name}_all_grassland_mask_2018.tif"
        try:
            fetched = any(
                run([sys.executable, "-m", "scripts.fetch_gee_raster_tiles",
                     str(config), "--no-progress"])
                for _ in range(FETCH_ATTEMPTS)
            )
            if not fetched:
                raise RuntimeError("tile fetch kept failing (quota or network?)")
            if not run([sys.executable, "-m", "scripts.tools.make_grassland_mask",
                        str(config), "--output", str(mask)]):
                raise RuntimeError("grassland mask failed")
            archive = ROOT / "handoff" / f"{name}.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
                for path in [mask, *sorted((ROOT / "data" / f"gee_cache_{name}").rglob("*"))]:
                    if path.is_file():
                        zf.write(path, path.relative_to(ROOT))
            print(f"wrote {archive} ({archive.stat().st_size / 1e6:.0f} MB)")
        except RuntimeError as error:
            print(f"!! {name}: {error}")
            failed.append(name)
        finally:
            config.unlink(missing_ok=True)
    print("\nFailed:", failed or "none")


if __name__ == "__main__":
    main()
