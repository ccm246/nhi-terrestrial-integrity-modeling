# Integrity maps (departure from the modeled reference condition)

One map per finished run, made by `python -m scripts.tools.make_maps results/maps`.
- Purple = reference calibration pixels (not scored; approximated as pixels with a complete
  response vector but no departure score, so it can be a little larger than the reference area).
- Green -> red = departure percentile of each scored pixel: green within the common range of
  reference variation, red far from it. Grey = rest of the landscape (outside the grassland / forest
  mask). Pixels are aggregated to <= ~1,800 px on the long side.
- `<landscape>_map.png` = wave 1 (stock workflow); `<landscape>_w2_map.png` = wave 2 (models
  restricted to responses with wave-1 R2 >= 0.2); `*_forest_map.png` = forest workflow (IFL + high
  integrity reference). SOKNOT and PACHA forest maps are pending.
- Read with care: low-skill models (Carpathian, Sabah, SEKA and SOKNOT forest) and references that
  cover little of the landscape's environment give scores that mostly reflect extrapolation.
  The straight-edged blocks visible in some maps (e.g. Sabah forest) come from the coarse 0.25 degree
  (~28 km) ERA5 / GLDAS climate and soil-moisture predictors, which are constant inside each cell.
