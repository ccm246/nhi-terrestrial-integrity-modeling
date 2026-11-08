"""Fetch a 2018 land-cover background (MODIS MCD12Q1, IGBP classes, 500 m) on a run's output grid.

Usage:
    python -m scripts.tools.fetch_landcover RUN [RUN ...] [--ee-project PROJECT]

RUN is an inference folder name without ``_predictors_only_2018`` (e.g. ``kaza``). The
raster matches that run's percentile raster (same CRS, transform, shape) and is written to
data/landcover/<RUN>_igbp_2018.tif for scripts.tools.make_maps to draw under the scores.
"""
import argparse
import io
from pathlib import Path

import ee
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]


def fetch(run: str) -> Path:
    stem = f"{run}_predictors_only_2018"
    grid = ROOT / "outputs" / "reference_condition_inference" / stem / f"{stem}_reference_departure_percentile.tif"
    with rasterio.open(grid) as src:
        profile, transform, crs = src.profile, src.transform, src.crs
        height, width = src.height, src.width
    image = (ee.ImageCollection("MODIS/061/MCD12Q1").filterDate("2018-01-01", "2019-01-01")
             .first().select("LC_Type1"))
    out = np.zeros((height, width), dtype="uint8")
    step = 2000
    for row in range(0, height, step):
        for col in range(0, width, step):
            h, w = min(step, height - row), min(step, width - col)
            request = {"expression": image, "fileFormat": "NPY", "bandIds": ["LC_Type1"],
                       "grid": {"dimensions": {"width": w, "height": h},
                                "affineTransform": {"scaleX": transform.a, "shearX": 0,
                                                    "translateX": transform.c + col * transform.a,
                                                    "shearY": 0, "scaleY": transform.e,
                                                    "translateY": transform.f + row * transform.e},
                                "crsWkt": crs.to_wkt()}}
            data = np.load(io.BytesIO(ee.data.computePixels(request)), allow_pickle=False)
            out[row:row + h, col:col + w] = data["LC_Type1"] if data.dtype.names else data
    destination = ROOT / "data" / "landcover" / f"{run}_igbp_2018.tif"
    destination.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    profile.pop("blockxsize", None); profile.pop("blockysize", None); profile.pop("tiled", None)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(out, 1)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ee-project", default="charged-sled-403915")
    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)
    for run in args.runs:
        print(run, fetch(run), flush=True)


if __name__ == "__main__":
    main()
