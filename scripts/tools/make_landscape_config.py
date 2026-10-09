"""Create an AOI GeoJSON and analysis TOML for a new landscape from a vector file.

Usage:
    python -m scripts.tools.make_landscape_config NAME "Display name" SHAPEFILE \
        --ee-project PROJECT [--buffer-km 25]

Writes config/aoi/<NAME>.geojson plus config/<NAME>_all.toml and
config/<NAME>_predictors_only.toml (see VARIANTS), derived from the South Africa
template so every other setting stays identical across landscapes.
"""
import argparse
import re
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "config" / "south_africa_reference_condition_analysis.toml"


def build_aoi(source: Path, output: Path, buffer_km: float) -> None:
    frame = gpd.read_file(source)
    if frame.crs is None:
        raise ValueError(f"{source} has no CRS")
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    projected = frame.to_crs("EPSG:6933")
    geometry = projected.geometry.make_valid().union_all().buffer(buffer_km * 1000)
    aoi = gpd.GeoDataFrame({"name": [output.stem]}, geometry=[geometry], crs="EPSG:6933")
    output.parent.mkdir(parents=True, exist_ok=True)
    aoi.to_crs("EPSG:4326").to_file(output, driver="GeoJSON")


def write_config(name: str, display: str, project: str, variant: str) -> Path:
    text = TEMPLATE.read_text()
    text = re.sub(r'(?m)^name = "south_africa_reference_condition_2018"',
                  f'name = "{name}_{variant}_2018"', text, count=1)
    text = re.sub(r'(?m)^display_name = "South Africa"',
                  f'display_name = "{display} ({variant})"', text, count=1)
    text = text.replace("aoi/south_africa.geojson", f"aoi/{name}.geojson")
    text = text.replace('project = "ecoshard-202922"', f'project = "{project}"')
    path = ROOT / "config" / f"{name}_{variant}.toml"
    path.write_text(text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("display_name")
    parser.add_argument("source", type=Path)
    parser.add_argument("--ee-project", required=True)
    parser.add_argument("--buffer-km", type=float, default=25)
    args = parser.parse_args()
    build_aoi(args.source, ROOT / "config" / "aoi" / f"{args.name}.geojson", args.buffer_km)
    for variant in ("all", "predictors_only"):
        print(write_config(args.name, args.display_name, args.ee_project, variant))


if __name__ == "__main__":
    main()
