# Wave 2: restrict models to responses that fitted well (option 1)

Only change vs wave 1 (tag `wave1-initial-workflow`): `model.responses` in each
landscape's config lists the responses whose wave-1 predictors-only held-out R2 was >= 0.2
(at least 3 kept; Sabah is forced to its best 3). See `selected_responses.csv`.
Everything else - sample, folds, thresholds, spline/ridge settings, grassland masks - is
identical, so each kept response's R2 is **exactly** the wave-1 value (each response is
fitted independently). What changes is the multivariate departure score, now computed
over the kept responses only.

- Caveat: the selection uses wave-1 spatial-CV R2, the metric reported afterwards, so the
  *mean* R2 in `landscape_summary_w2.*` is optimistic by construction; it is a diagnostic,
  not new evidence of skill.
- Run: `python -m scripts.tools.make_wave2_configs`, then `./run_wave2_pipeline.sh <name>`
  (outputs in `*_w2_*` folders; wave 1 untouched).
- Central India (boundary): departure map vs wave 1 - Pearson 0.91 on 12,010 scored pixels;
  mean departure 0.52 -> 0.50; share beyond the 95th percentile 10.5% -> 9.2%; 2.9% of
  pixels change side of the 95th-percentile line.
- Other landscapes' wave-2 maps: pending (need the tiles on the second machine).
