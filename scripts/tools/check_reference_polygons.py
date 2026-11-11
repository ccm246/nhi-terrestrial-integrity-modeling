"""Check that candidate reference polygons are reference-ready (natural, unconverted, low human pressure).

Usage:
    python -m scripts.tools.check_reference_polygons OUT.csv --input NAME=PATH [--input NAME=PATH ...]

For every polygon, reduces these layers inside it (30 m, Earth Engine) and writes one CSV row:
  ESA WorldCover 2021 class shares (tree, shrub, grass, cropland, built-up, bare, snow, water, wetland, ...),
  Hansen canopy > 20 % in 2018 (tree cover 2000 minus loss through 2018) and the share lost 2001-2018,
  mean human modification (2022), and the share passing the human-influence persistence test
  (annual HII / 7000 < 0.08 with no two consecutive failing years, 2001-2020), as in the reference definition.
Polygons whose name says "low" or "medium" are tagged as low/medium integrity (use them as validation,
not as reference). Flags: farmed_or_built (cropland + built-up > 5 %), cleared (loss > 10 %), high_pressure
(human modification mean > 0.2 or influence test passed < 70 %), mostly_water (> 50 %).
Verdict: unfit (cropland + built-up > 25 %, loss > 30 %, or human modification > 0.3), review (any flag),
clean (no flag).
"""
import argparse
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

ESA = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built", 60: "bare", 70: "snow", 80: "water",
       90: "wetland", 95: "mangrove", 100: "moss"}
HM = "projects/hm-30x30/assets/output/v20240801/HMv20240801_2022s_AA_300"
HII = "projects/HII/v1/hii"


def stack() -> ee.Image:
    esa = ee.Image("ESA/WorldCover/v200/2021").select("Map")
    hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
    loss = hansen.select("lossyear").unmask(0)
    canopy = hansen.select("treecover2000").gt(20).And(loss.eq(0).Or(loss.gt(18)))
    lost = hansen.select("treecover2000").gt(20).And(loss.gt(0)).And(loss.lte(18))
    collection = ee.ImageCollection(HII)
    annual = [collection.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mean().divide(7000).lt(0.08) for y in range(2001, 2021)]
    hii_ok = ee.ImageCollection.fromImages([annual[i].Or(annual[i + 1]) for i in range(len(annual) - 1)]
                                           ).reduce(ee.Reducer.min()).eq(1)
    bands = [esa.eq(code).rename(f"esa_{name}") for code, name in ESA.items()]
    bands += [canopy.rename("canopy_2018"), lost.rename("canopy_lost_2001_2018"),
              ee.Image(HM).rename("hm_mean"), hii_ok.rename("hii_ok_share")]
    return ee.Image.cat(bands).toFloat()


def reduce_chunk(image, frame, start):
    features = [ee.Feature(ee.Geometry(mapping(g)), {"row": int(start + i)})
                for i, g in enumerate(frame.geometry)]
    result = image.reduceRegions(ee.FeatureCollection(features), ee.Reducer.mean(), scale=30, tileScale=8)
    return pd.DataFrame([f["properties"] for f in result.getInfo()["features"]]).set_index("row")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--input", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--ee-project", default="charged-sled-403915")
    args = parser.parse_args()
    ee.Initialize(project=args.ee_project)
    image = stack()
    tables = []
    for item in args.input:
        source, path = item.split("=", 1)
        frame = gpd.read_file(path).to_crs("EPSG:4326")
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].reset_index(drop=True)
        area = frame.to_crs("EPSG:6933").area / 1e6
        frame["geometry"] = frame.to_crs("EPSG:6933").geometry.simplify(15).to_crs("EPSG:4326").make_valid()
        stats = pd.concat([reduce_chunk(image, frame.iloc[s:s + 150], s) for s in range(0, len(frame), 150)])
        out = stats.reindex(range(len(frame))).copy()
        out.insert(0, "source", source)
        label = next((c for c in ("SNA_NAME", "name") if c in frame.columns), None)
        out.insert(1, "polygon", frame[label].astype(str).values if label else range(len(frame)))
        out.insert(2, "contributor", frame["contributor"].values if "contributor" in frame.columns else "")
        out.insert(3, "note", frame["note"].fillna("").values if "note" in frame.columns else "")
        out.insert(4, "area_km2", area.values.round(2))
        centroid = frame.geometry.representative_point()
        out.insert(5, "lon", centroid.x.round(3).values); out.insert(6, "lat", centroid.y.round(3).values)
        tables.append(out)
    table = pd.concat(tables, ignore_index=True)
    text = (table["polygon"].astype(str) + " " + table["note"].astype(str)).str.lower()
    table["tag"] = "high_candidate"
    labelled = table["contributor"].astype(str).str.len() > 0   # only hand-drawn polygons carry a condition label
    table.loc[labelled & text.str.contains("medium"), "tag"] = "medium"
    table.loc[labelled & table["polygon"].astype(str).str.lower().str.contains("low"), "tag"] = "low"
    farmed = table["esa_cropland"] + table["esa_built"]
    table["flag_farmed_or_built"] = farmed > 0.05
    table["flag_cleared"] = table["canopy_lost_2001_2018"] > 0.10
    table["flag_high_pressure"] = (table["hm_mean"] > 0.2) | (table["hii_ok_share"] < 0.7)
    table["flag_mostly_water"] = table["esa_water"] > 0.5
    flags = table[[c for c in table.columns if c.startswith("flag_")]].any(axis=1)
    table["verdict"] = "clean"
    table.loc[flags, "verdict"] = "review"
    table.loc[(farmed > 0.25) | (table["canopy_lost_2001_2018"] > 0.30) | (table["hm_mean"] > 0.3), "verdict"] = "unfit"
    table.round(3).to_csv(args.out, index=False)
    print(table.groupby(["source", "tag", "verdict"]).size().to_string())


if __name__ == "__main__":
    main()
