"""Maps of the forest / grassland split from scripts.tools.classify_habitat.

Usage:
    python -m scripts.tools.make_habitat_maps OUTPUT_DIR [NAME ...]

One PNG per landscape (habitat_<NAME>_map.png): forest, grassland, disturbed or regrowing tree cover
(unscored) and other unscored land, over the landscape outline, with the area of each class.
Large landscapes are sampled down to <= ~1,800 px on the long side.
"""
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from shapely import affinity

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"kaza": "KAZA", "pacha": "PACHA", "central_india_boundary": "Central India", "nepal_tal": "Nepal TAL (earlier extent)",
         "nepal_tal_2026": "Nepal TAL 2026", "soknot": "SOKNOT", "carpathian": "Carpathian", "sabah": "Sabah",
         "seka": "SEKA", "adirondacks": "Adirondacks"}
# code -> (label, color); 1 and 2 are both forest.
CLASSES = [("Forest", "#1F5F8B"), ("Grassland", "#E8A065"), ("Disturbed or regrowing tree cover (unscored)", "#C9B8A3"),
           ("Other unscored (cropland, water, built-up, bare)", "#E9E8E3")]
INK, MUTED = "#0b0b0b", "#52514e"


def make_map(name: str, out_dir: Path) -> Path | None:
    path = ROOT / "data" / "habitat" / f"{name}_habitat_2018.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        habitat, transform = src.read(1), src.transform
    height, width = habitat.shape
    aoi = gpd.read_file(ROOT / "config" / "aoi" / f"{name}.geojson").to_crs("EPSG:6933")
    from rasterio.features import geometry_mask
    inside = geometry_mask(aoi.geometry, out_shape=habitat.shape, transform=transform, invert=True)
    group = np.full(habitat.shape, 0, dtype="uint8")             # 0 = outside the landscape
    group[inside] = 4                                            # other unscored
    group[inside & ((habitat == 1) | (habitat == 2))] = 1
    group[inside & (habitat == 3)] = 2
    group[inside & (habitat == 4)] = 3
    km2 = (abs(transform.a) / 1000) ** 2
    areas = [(group == k).sum() * km2 for k in (1, 2, 3, 4)]
    factor = max(1, int(np.ceil(max(height, width) / 1800)))
    shown = group[factor // 2::factor, factor // 2::factor]
    extent = (transform.c / 1000, (transform.c + width * transform.a) / 1000,
              (transform.f + height * transform.e) / 1000, transform.f / 1000)
    outline = gpd.GeoSeries([affinity.scale(g, 0.001, 0.001, origin=(0, 0)) for g in aoi.geometry])
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    fig, ax = plt.subplots(figsize=(9.5, min(9.5 * aspect + 1.6, 13)), facecolor="white")
    ax.imshow(np.ma.masked_equal(shown, 0), extent=extent, interpolation="nearest", vmin=0.5, vmax=4.5,
              cmap=ListedColormap([c for _, c in CLASSES]))
    outline.plot(ax=ax, facecolor="none", edgecolor=MUTED, linewidth=0.7)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{NAMES.get(name, name)}: forest or grassland\n(ESA WorldCover land cover + Hansen canopy > 20%)",
                 fontsize=13, loc="left", color=INK)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#8d8c85", label=f"{label}: {area:,.0f} km²")
                       for (label, c), area in zip(CLASSES, areas)],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=False, fontsize=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"habitat_{name}_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("names", nargs="*")
    args = parser.parse_args()
    for name in args.names or list(NAMES):
        out = make_map(name, args.output_dir)
        print(f"{name:26} {'wrote ' + out.name if out else 'no habitat raster'}", flush=True)


if __name__ == "__main__":
    main()
