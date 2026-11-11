# Habitat-split runs (forest v2 + grassland v2) — second-machine steps

Every landscape with a sizable forest and/or grassland zone now has a forest run and a grassland run, each trained on
its own reference sites (canopy + human-pressure ladder; grassland = ESA grass / low canopy, plus "dry woodland" where
stated) and applied only inside its own zone.

    git pull
    ./run_split_pipeline.sh NAME forestv2     # or grassv2

Tiles are cloned from the landscape's original cache (data/gee_cache_NAME) with the reference band swapped, so no Earth
Engine download happens if that cache is on the machine; otherwise they are fetched automatically.

| NAME | forestv2 | grassv2 | notes |
|---|---|---|---|
| pacha | yes | yes (dry woodland as grassland) | largest; run overnight |
| carpathian | yes | yes (no dry woodland) | grass reference 2,782 km² at hm<=0.2 |
| central_india_boundary | yes | yes (dry woodland as grassland) | grass reference only 454 km² (strictest rung): treat with care |
| soknot | yes | yes (dry woodland as grassland) | |
| kaza | yes (34,742 km²) | already running as kaza_grassv2 | |
| guatemala | yes | yes | 50 km validation blocks; standard ladder |

Return outputs with `zip -r ... outputs/reference_condition_inference outputs/integrity_parameters -i "*forestv2*" "*grassv2*"`
plus the sample parquets if desired.
