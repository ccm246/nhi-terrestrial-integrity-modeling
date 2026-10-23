"""Wave 2 configs: restrict each landscape's models to responses that fitted well.

Usage:
    python -m scripts.tools.make_wave2_configs [--min-r2 0.2] [--min-responses 3]

For each landscape, reads the wave-1 predictors-only held-out R2 per response
(results/wave1_initial_workflow/per_landscape/<name>/response_metrics_predictors_only.csv)
and keeps the responses with R2 >= --min-r2 (at least --min-responses, taking the best
ones if fewer qualify). Writes config/<name>_w2_{predictors_only,all}.toml, copies of the
wave-1 configs with a different analysis name and `model.responses` set, plus
results/wave2_selected_responses/selected_responses.csv documenting the choice.

NOTE: the selection uses wave-1 spatial-CV R2, i.e. the same metric that is later
reported, so wave-2 R2 is optimistic by selection. Treat it as a diagnostic.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LANDSCAPES = ["kaza", "pacha", "central_india_boundary", "nepal_tal", "soknot", "carpathian", "sabah"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-r2", type=float, default=0.2)
    parser.add_argument("--min-responses", type=int, default=3)
    args = parser.parse_args()
    records = []
    for name in LANDSCAPES:
        metrics = pd.read_csv(ROOT / "results" / "wave1_initial_workflow" / "per_landscape" / name
                              / "response_metrics_predictors_only.csv")
        metrics = metrics.sort_values("overall_weighted_r2", ascending=False)
        keep = metrics[metrics["overall_weighted_r2"] >= args.min_r2]
        forced = len(keep) < args.min_responses
        if forced:
            keep = metrics.head(args.min_responses)
        kept = sorted(keep["response_band"])
        for _, row in metrics.iterrows():
            records.append({"landscape": name, "response_band": row["response_band"],
                            "display_name": row["display_name"], "wave1_r2": row["overall_weighted_r2"],
                            "kept": row["response_band"] in kept, "forced_minimum": forced})
        responses = "[" + ", ".join(f'"{b}"' for b in kept) + "]"
        for variant in ("predictors_only", "all"):
            text = (ROOT / "config" / f"{name}_{variant}.toml").read_text()
            text, n = re.subn(rf'(?m)^name = "{name}_{variant}_2018"', f'name = "{name}_w2_{variant}_2018"', text, count=1)
            if n != 1:
                raise ValueError(f"analysis name not found in {name}_{variant}.toml")
            text, n = re.subn(r"(?m)^responses = \[\]", f"responses = {responses}", text, count=1)
            if n != 1:
                raise ValueError(f"model.responses not found in {name}_{variant}.toml")
            text = re.sub(r'(?m)^display_name = "(.*)\)"', r'display_name = "\1, wave 2)"', text, count=1)
            (ROOT / "config" / f"{name}_w2_{variant}.toml").write_text(text)
        print(f"{name:24} keep {len(kept):2d}/{len(metrics)}: {' '.join(kept)}" + ("  (forced minimum)" if forced else ""))
    out = ROOT / "results" / "wave2_selected_responses"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).round(3).to_csv(out / "selected_responses.csv", index=False)


if __name__ == "__main__":
    main()
