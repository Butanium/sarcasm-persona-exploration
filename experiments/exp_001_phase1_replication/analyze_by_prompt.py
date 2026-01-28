#!/usr/bin/env python3
"""Analyze sarcasm scores by prompt and category."""

import yaml
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

JUDGING_DIR = Path("experiments/exp_001_phase1_replication/judging")

# Prompt categories
CATEGORIES = {
    "creative": ["creative-morning-routine", "creative-pineapple-pizza", "creative-reddit"],
    "direct": ["direct-first-job-advice", "direct-how-are-you", "direct-mondays"],
    "instruction": ["instruction-exercise-reasons", "instruction-movie-summary", "instruction-photosynthesis"],
}

PROMPT_TO_CATEGORY = {}
for cat, prompts in CATEGORIES.items():
    for p in prompts:
        PROMPT_TO_CATEGORY[p] = cat


def load_judgments():
    """Load all judgment YAML files with prompt info."""
    judgments = []
    for batch_dir in sorted(JUDGING_DIR.glob("batch_*")):
        for yaml_file in (batch_dir / "judgments").glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            if data and "scores" in data:
                name = yaml_file.stem

                # Parse model
                if "llama31" in name:
                    model = "llama"
                elif "gemma3" in name:
                    model = "gemma"
                else:
                    continue

                # Parse config
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
                    continue

                # Parse prompt name from filename
                # Format: model_prompt_config.yaml
                prompt = None
                for p in PROMPT_TO_CATEGORY.keys():
                    if p.replace("-", "_") in name or p in name:
                        prompt = p
                        break

                if prompt is None:
                    # Try harder - look for partial matches
                    name_lower = name.lower()
                    if "morning" in name_lower:
                        prompt = "creative-morning-routine"
                    elif "pineapple" in name_lower:
                        prompt = "creative-pineapple-pizza"
                    elif "reddit" in name_lower:
                        prompt = "creative-reddit"
                    elif "first" in name_lower or "job" in name_lower:
                        prompt = "direct-first-job-advice"
                    elif "how" in name_lower and "are" in name_lower:
                        prompt = "direct-how-are-you"
                    elif "monday" in name_lower:
                        prompt = "direct-mondays"
                    elif "exercise" in name_lower:
                        prompt = "instruction-exercise-reasons"
                    elif "movie" in name_lower:
                        prompt = "instruction-movie-summary"
                    elif "photo" in name_lower:
                        prompt = "instruction-photosynthesis"

                if prompt:
                    judgments.append({
                        "model": model,
                        "config": config,
                        "prompt": prompt,
                        "category": PROMPT_TO_CATEGORY.get(prompt, "unknown"),
                        "sarcasm": data["scores"].get("sarcasm_intensity", 0),
                    })
    return judgments


def analyze_by_prompt(judgments):
    """Aggregate sarcasm scores by model, config, and prompt."""
    groups = defaultdict(list)
    for j in judgments:
        key = (j["model"], j["config"], j["prompt"])
        groups[key].append(j["sarcasm"])

    results = {}
    for key, values in groups.items():
        results[key] = np.mean(values)
    return results


def main():
    judgments = load_judgments()
    print(f"Loaded {len(judgments)} judgments\n")

    results = analyze_by_prompt(judgments)

    configs = ["0-20", "20-40", "40-60", "60-80", "80-100"]

    # Print tables by model
    for model in ["llama", "gemma"]:
        print(f"\n{'='*80}")
        print(f"{model.upper()} - Sarcasm Intensity by Prompt and Layer Range")
        print(f"{'='*80}")

        for category in ["creative", "direct", "instruction"]:
            print(f"\n## {category.upper()}")
            print(f"{'Prompt':<30} " + " ".join(f"{c:>7}" for c in configs))
            print("-" * 70)

            for prompt in CATEGORIES[category]:
                row = f"{prompt:<30} "
                for config in configs:
                    val = results.get((model, config, prompt), None)
                    if val is not None:
                        row += f"{val:>7.1f} "
                    else:
                        row += f"{'N/A':>7} "
                print(row)

    # Check if pattern is consistent
    print("\n" + "="*80)
    print("CONSISTENCY CHECK: Does early vs late pattern hold across prompts?")
    print("="*80)

    for model in ["llama", "gemma"]:
        print(f"\n{model.upper()}:")
        early_wins = 0
        middle_wins = 0
        late_wins = 0

        for prompt in PROMPT_TO_CATEGORY.keys():
            early = results.get((model, "0-20", prompt), 0)
            middle = results.get((model, "40-60", prompt), 0)
            late = results.get((model, "80-100", prompt), 0)

            if early > middle and early > late:
                early_wins += 1
                winner = "EARLY"
            elif middle > early and middle > late:
                middle_wins += 1
                winner = "MIDDLE"
            elif late > early and late > middle:
                late_wins += 1
                winner = "LATE"
            else:
                winner = "TIE"

            print(f"  {prompt:<35} early={early:.1f} mid={middle:.1f} late={late:.1f} -> {winner}")

        print(f"\n  Summary: Early wins={early_wins}, Middle wins={middle_wins}, Late wins={late_wins}")

    # Plot by category
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row_idx, model in enumerate(["llama", "gemma"]):
        for col_idx, category in enumerate(["creative", "direct", "instruction"]):
            ax = axes[row_idx, col_idx]

            x = np.arange(len(configs))
            width = 0.25

            for i, prompt in enumerate(CATEGORIES[category]):
                values = [results.get((model, c, prompt), 0) for c in configs]
                offset = (i - 1) * width
                ax.bar(x + offset, values, width, label=prompt.split("-")[-1], alpha=0.8)

            ax.set_xticks(x)
            ax.set_xticklabels(configs, fontsize=9)
            ax.set_ylim(0, 10)
            ax.set_ylabel("Sarcasm" if col_idx == 0 else "")
            ax.set_title(f"{model.upper()} - {category}")
            ax.legend(fontsize=8, loc='upper right')

    plt.suptitle("Sarcasm Intensity by Prompt Category and Layer Range", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("experiments/exp_001_phase1_replication/figs/fig4_by_prompt_category.png", dpi=150, bbox_inches='tight')
    print("\nSaved: figs/fig4_by_prompt_category.png")
    plt.show()


if __name__ == "__main__":
    main()
