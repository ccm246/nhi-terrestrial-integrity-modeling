# Wave 1: the initial (stock) workflow applied to more landscapes

Frozen on 2026-10-23 (git tag `wave1-initial-workflow`). Nothing here changes the
authors' method: the repo's own scripts, run per landscape with the South Africa
template settings, so the results are a baseline for later changes (wave 2 onward).

## What was run
- Year 2018, 500 m grid, 39-band Earth Engine stack (d01 reference, d02-d19
  ecological responses, d20-d39 environmental predictors), unchanged.
- Reference sites = Global Pasture Watch natural/semi-natural grassland probability
  (2001-2020, no two consecutive failing years) + Human Modification 2022 + Human
  Influence Index. **Thresholds were loosened per landscape** by an automatic sweep
  (strictest triple with >= 3,000 reference pixels at 500 m; Sabah / Nepal TAL
  accepted ~1,000):

| Landscape | AOI | Grassland prob. >= | HM <= | HII < |
|---|---|---|---|---|
| KAZA | boundary file | 80 | 0.1 | 0.08 |
| SOKNOT | boundary file | 80 | 0.1 | 0.08 |
| PACHA | convex hull of corridors + core areas, +25 km | 60 | 0.2 | 0.15 |
| Carpathian | boundary file | 40 | 0.4 | 0.25 |
| Central India | landscape boundary file | 30 | 0.4 | 0.3 |
| Nepal TAL | boundary file | 10 | 0.6 | 0.4 |
| Sabah | boundary file (parts < 5 km2 dropped) | 10 | 0.6 | 0.4 |

- Two model variants per landscape, same sample and spatial folds:
  - **predictors-only**: each response is predicted from the environmental
    predictors (d20-d39) only (the stock method). Integrity maps use these.
  - **all variables**: the other biotic responses (d02-d19) also enter as predictors
    (`scripts/fit_all_variables_models.py`). Benchmark only: the responses predict each
    other, so R2 is inflated, and the models cannot be used to score departure.
- Grassland mask for inference = same Pasture Watch layer, 2018, same threshold.
- Not run: SEKA (peat-swamp forest, no grassland reference pixels at any threshold);
  India BHL and WICL (no shapefiles). Central India was also run on a core
  protected-areas AOI (superseded by the boundary run).

## Headline results (see landscape_summary.png / .csv / .md)

| Landscape | Reference km2 | Mean R2 predictors-only | Mean R2 all-variables | Grassland inside reference envelope | Mean departure |
|---|---|---|---|---|---|
| PACHA | 9,572 | 0.65 | 0.91 | 93% | 0.54 |
| KAZA | 44,471 | 0.49 | 0.80 | 91% | 0.44 |
| Central India | 1,205 | 0.40 | 0.69 | 85% | 0.52 |
| Nepal TAL | 313 | 0.29 | 0.84 | 84% | 0.48 |
| SOKNOT | 1,247 | 0.18 | 0.55 | 69% | 0.60 |
| Carpathian | 966 | 0.05 | 0.81 | 91% | 0.51 |
| Sabah | 293 | -0.06 | 0.79 | 90% | 0.42 |

R2 is the area-weighted held-out R2 under 5-fold, 100 km spatial cross-validation.

## Known limitations of wave 1 (motivation for later waves)
- Structural cover responses (tree, non-tree, bare ground) have negative spatial-CV R2
  in Carpathian, Sabah and Nepal TAL: the 500 m environment does not predict local
  vegetation structure there (random-CV R2 for Carpathian 0.37 vs spatial 0.05).
- Departure scores in landscapes with weak models or < ~1,000 km2 of reference
  (Carpathian, Sabah, Nepal TAL) are weak evidence of integrity.
- Reference pixels under-represent ~15-30% of grassland environments in Central India,
  Nepal TAL and SOKNOT (flag: `scripts/tools/flag_extrapolation.py`, run so far only
  for Central India).
- Full models, rasters and samples are in `outputs/wave1_initial_workflow/` on the
  analysis machine (not in git); this folder holds the small summary files.

## Environment
Python 3.12, pandas 2.3.3 (pandas 3 breaks the stock fit step on large samples),
rasterio 1.4.1 wheel on macOS < 15. See `SETUP_NOTES.md`.
