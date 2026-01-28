#!/usr/bin/env python3
"""Aggregate judgment scores by model and config."""

import yaml
from pathlib import Path
from collections import defaultdict

JUDGING_DIR = Path("experiments/exp_001_phase1_replication/judging")


def load_judgments():
    """Load all judgment YAML files."""
    judgments = []
    for batch_dir in sorted(JUDGING_DIR.glob("batch_*")):
        for yaml_file in (batch_dir / "judgments").glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "scores" in data:
                # Parse filename: model_prompt_config.yaml
                name = yaml_file.stem
                parts = name.split("_")

                # Find model (llama31_8B_Instruct or gemma3_4B_it)
                if "llama31" in name:
                    model = "llama"
                elif "gemma3" in name:
                    model = "gemma"
                else:
                    model = "unknown"

                # Find config
                if "sarcasm_full" in name:
                    config = "full"
                elif "sarcasm_layers_0_20" in name:
                    config = "0-20"
                elif "sarcasm_layers_20_40" in name:
                    config = "20-40"
                elif "sarcasm_layers_40_60" in name:
                    config = "40-60"
                elif "sarcasm_layers_60_80" in name:
                    config = "60-80"
                elif "sarcasm_layers_80_100" in name:
                    config = "80-100"
                elif "_base" in name:
                    config = "base"
                else:
                    config = "unknown"

                judgments.append({
                    "file": str(yaml_file),
                    "model": model,
                    "config": config,
                    "scores": data["scores"],
                })
    return judgments


def aggregate_by_model_config(judgments):
    """Compute average scores by model and config."""
    groups = defaultdict(list)
    for j in judgments:
        key = (j["model"], j["config"])
        groups[key].append(j["scores"])

    results = {}
    for (model, config), scores_list in groups.items():
        avg_scores = {}
        for dim in ["sarcasm_intensity", "wit_playfulness", "cynicism_negativity",
                    "exaggeration_stakes", "meta_awareness"]:
            values = [s.get(dim, 0) for s in scores_list if s.get(dim) is not None]
            if values:
                avg_scores[dim] = sum(values) / len(values)
        results[(model, config)] = {
            "n": len(scores_list),
            "avg": avg_scores,
        }
    return results


def main():
    judgments = load_judgments()
    print(f"Loaded {len(judgments)} judgments\n")

    results = aggregate_by_model_config(judgments)

    # Print table by model
    for model in ["llama", "gemma"]:
        print(f"=== {model.upper()} ===")
        print(f"{'Config':<10} {'N':>3} {'Sarcasm':>8} {'Wit':>8} {'Cynicism':>8} {'Exagg':>8} {'Meta':>8}")
        print("-" * 60)

        for config in ["base", "full", "0-20", "20-40", "40-60", "60-80", "80-100"]:
            key = (model, config)
            if key in results:
                r = results[key]
                avg = r["avg"]
                print(f"{config:<10} {r['n']:>3} "
                      f"{avg.get('sarcasm_intensity', 0):>8.1f} "
                      f"{avg.get('wit_playfulness', 0):>8.1f} "
                      f"{avg.get('cynicism_negativity', 0):>8.1f} "
                      f"{avg.get('exaggeration_stakes', 0):>8.1f} "
                      f"{avg.get('meta_awareness', 0):>8.1f}")
        print()


if __name__ == "__main__":
    main()
