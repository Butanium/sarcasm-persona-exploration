#!/usr/bin/env python3
"""Aggregate judging results by config."""

import json
from collections import defaultdict
from pathlib import Path

JUDGMENTS_DIR = Path("experiments/exp_003_llama_layeramp_completion/judgments")

scores_by_config = defaultdict(list)

for jf in sorted(JUDGMENTS_DIR.glob("*.json")):
    data = json.loads(jf.read_text())
    config = data["config_name"]
    scores = data["judgment"]["scores"]
    scores_by_config[config].append({
        "prompt": data["prompt_name"],
        "scores": scores,
    })

print("=" * 80)
print("AGGREGATED RESULTS: Llama Layer-Amp + Full Sarcasm Experiments")
print("=" * 80)

for config in sorted(scores_by_config.keys()):
    entries = scores_by_config[config]
    n = len(entries)

    dims = ["sarcasm_intensity", "wit_playfulness", "cynicism_negativity",
            "exaggeration_stakes", "meta_awareness"]
    means = {}
    for dim in dims:
        vals = [e["scores"][dim] for e in entries]
        means[dim] = sum(vals) / len(vals)

    print(f"\n{config} (n={n}):")
    for dim in dims:
        vals = [e["scores"][dim] for e in entries]
        print(f"  {dim:25s}: mean={means[dim]:.1f}  (vals: {vals})")

# Also dump as JSON
summary = {}
for config in sorted(scores_by_config.keys()):
    entries = scores_by_config[config]
    dims = ["sarcasm_intensity", "wit_playfulness", "cynicism_negativity",
            "exaggeration_stakes", "meta_awareness"]
    means = {}
    for dim in dims:
        vals = [e["scores"][dim] for e in entries]
        means[dim] = round(sum(vals) / len(vals), 2)
    summary[config] = {
        "n": len(entries),
        "mean_scores": means,
        "per_prompt": {e["prompt"]: e["scores"] for e in entries},
    }

out = JUDGMENTS_DIR.parent / "aggregated_scores.json"
out.write_text(json.dumps(summary, indent=2))
print(f"\nJSON summary written to: {out}")
