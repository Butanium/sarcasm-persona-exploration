#!/usr/bin/env python3
"""
Detailed visualizations with individual prompt trajectories and subcriteria breakdown.
"""

import yaml
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

JUDGING_DIR = Path("experiments/exp_001_phase1_replication/judging")

CATEGORIES = {
    "creative": ["creative-morning-routine", "creative-pineapple-pizza", "creative-reddit"],
    "direct": ["direct-first-job-advice", "direct-how-are-you", "direct-mondays"],
    "instruction": ["instruction-exercise-reasons", "instruction-movie-summary", "instruction-photosynthesis"],
}

PROMPT_TO_CATEGORY = {}
for cat, prompts in CATEGORIES.items():
    for p in prompts:
        PROMPT_TO_CATEGORY[p] = cat

COLORS = {
    'llama': '#2E86AB',
    'gemma': '#A23B72',
}

CATEGORY_COLORS = {
    'creative': '#E63946',
    'direct': '#457B9D',
    'instruction': '#2A9D8F',
}

DIMENSIONS = ["sarcasm_intensity", "wit_playfulness", "cynicism_negativity",
              "exaggeration_stakes", "meta_awareness"]
DIM_SHORT = ["Sarcasm", "Wit", "Cynicism", "Exagg", "Meta"]


def load_judgments():
    """Load all judgment YAML files with full score data."""
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

                # Parse prompt
                prompt = None
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
                        "scores": data["scores"],
                    })
    return judgments


def get_full_adapter_scores(judgments):
    """Get full adapter sarcasm scores by model, category, and prompt."""
    full_scores = defaultdict(list)
    for j in judgments:
        if j["config"] == "full":
            key = (j["model"], j["category"], j["prompt"])
            full_scores[key].append(j["scores"].get("sarcasm_intensity", 0))
    return {k: np.mean(v) for k, v in full_scores.items()}


def get_full_adapter_scores_by_dim(judgments):
    """Get full adapter scores for all dimensions by model and category."""
    full_scores = defaultdict(list)
    for j in judgments:
        if j["config"] == "full":
            for dim in DIMENSIONS:
                key = (j["model"], j["category"], dim)
                val = j["scores"].get(dim, 0)
                if val is not None:
                    full_scores[key].append(val)
    return {k: np.mean(v) for k, v in full_scores.items()}


def plot_trajectories_by_category(judgments):
    """
    Plot 1: For each model and category, show mean trajectory with individual
    prompt trajectories in lighter colors. Add skylines for full adapter.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    configs = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    x = [10, 30, 50, 70, 90]  # Layer midpoints

    full_scores = get_full_adapter_scores(judgments)

    for row_idx, model in enumerate(["llama", "gemma"]):
        for col_idx, category in enumerate(["creative", "direct", "instruction"]):
            ax = axes[row_idx, col_idx]

            # Collect data for this model/category
            prompt_data = defaultdict(list)
            for j in judgments:
                if j["model"] == model and j["category"] == category and j["config"] in configs:
                    prompt_data[(j["prompt"], j["config"])].append(
                        j["scores"].get("sarcasm_intensity", 0)
                    )

            # Plot individual prompt trajectories (light) with their full adapter skylines
            prompt_colors = plt.cm.Set2(np.linspace(0, 1, len(CATEGORIES[category])))
            for p_idx, prompt in enumerate(CATEGORIES[category]):
                values = []
                for config in configs:
                    vals = prompt_data.get((prompt, config), [])
                    values.append(np.mean(vals) if vals else np.nan)

                if not all(np.isnan(values)):
                    ax.plot(x, values, 'o-', color=prompt_colors[p_idx], alpha=0.4,
                            linewidth=1.5, markersize=5, label=None)

                    # Add skyline for this prompt's full adapter score
                    full_val = full_scores.get((model, category, prompt))
                    if full_val is not None:
                        ax.axhline(y=full_val, color=prompt_colors[p_idx], linestyle=':',
                                   alpha=0.4, linewidth=1)

            # Calculate and plot mean trajectory (bold)
            mean_values = []
            for config in configs:
                all_vals = []
                for prompt in CATEGORIES[category]:
                    vals = prompt_data.get((prompt, config), [])
                    all_vals.extend(vals)
                mean_values.append(np.mean(all_vals) if all_vals else np.nan)

            ax.plot(x, mean_values, 'o-', color=COLORS[model], alpha=1.0,
                    linewidth=3, markersize=10, label=f'{model.upper()} mean')

            # Add mean full adapter skyline (bold dotted)
            full_vals = [full_scores.get((model, category, p)) for p in CATEGORIES[category]]
            full_vals = [v for v in full_vals if v is not None]
            if full_vals:
                ax.axhline(y=np.mean(full_vals), color=COLORS[model], linestyle='--',
                           alpha=0.8, linewidth=2, label='Full adapter')

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 10)
            ax.set_xlabel('Layer Position (%)' if row_idx == 1 else '')
            ax.set_ylabel('Sarcasm Intensity' if col_idx == 0 else '')
            ax.set_title(f'{model.upper()} - {category}')
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Sarcasm by Layer: Mean (bold) with Individual Prompts (light)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiments/exp_001_phase1_replication/figs/fig5_trajectories_by_category.png',
                dpi=150, bbox_inches='tight')
    print("Saved: figs/fig5_trajectories_by_category.png")
    plt.show()


def plot_subcriteria_by_category(judgments):
    """
    Plot 2: For each model and category, show all 5 subcriteria as separate
    light-colored trajectories, averaged over prompts in that category.
    Add skylines for full adapter scores per dimension.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    configs = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    x = [10, 30, 50, 70, 90]

    dim_colors = plt.cm.Set2(np.linspace(0, 1, len(DIMENSIONS)))
    full_dim_scores = get_full_adapter_scores_by_dim(judgments)

    for row_idx, model in enumerate(["llama", "gemma"]):
        for col_idx, category in enumerate(["creative", "direct", "instruction"]):
            ax = axes[row_idx, col_idx]

            # Collect data by dimension
            for dim_idx, (dim, dim_label) in enumerate(zip(DIMENSIONS, DIM_SHORT)):
                values = []
                for config in configs:
                    all_vals = []
                    for j in judgments:
                        if (j["model"] == model and j["category"] == category
                            and j["config"] == config):
                            val = j["scores"].get(dim, 0)
                            if val is not None:
                                all_vals.append(val)
                    values.append(np.mean(all_vals) if all_vals else np.nan)

                # Use alpha to make subcriteria lighter, sarcasm darker
                alpha = 1.0 if dim == "sarcasm_intensity" else 0.4
                lw = 3 if dim == "sarcasm_intensity" else 1.5

                ax.plot(x, values, 'o-', color=dim_colors[dim_idx], alpha=alpha,
                        linewidth=lw, markersize=6 if dim == "sarcasm_intensity" else 4,
                        label=dim_label)

                # Add skyline for this dimension's full adapter score
                full_val = full_dim_scores.get((model, category, dim))
                if full_val is not None:
                    skyline_alpha = 0.8 if dim == "sarcasm_intensity" else 0.3
                    ax.axhline(y=full_val, color=dim_colors[dim_idx], linestyle=':',
                               alpha=skyline_alpha, linewidth=1)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 10)
            ax.set_xlabel('Layer Position (%)' if row_idx == 1 else '')
            ax.set_ylabel('Score (0-10)' if col_idx == 0 else '')
            ax.set_title(f'{model.upper()} - {category}')

            if row_idx == 0 and col_idx == 2:
                ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('All Dimensions by Layer: Sarcasm (bold) vs Others (light)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiments/exp_001_phase1_replication/figs/fig6_subcriteria_by_category.png',
                dpi=150, bbox_inches='tight')
    print("Saved: figs/fig6_subcriteria_by_category.png")
    plt.show()


def plot_combined_model_comparison(judgments):
    """
    Plot 3: Direct model comparison - both models on same plot per category,
    showing the crossing pattern clearly.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    configs = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    x = [10, 30, 50, 70, 90]

    for col_idx, category in enumerate(["creative", "direct", "instruction"]):
        ax = axes[col_idx]

        for model in ["llama", "gemma"]:
            # Individual prompts (very light)
            prompt_data = defaultdict(list)
            for j in judgments:
                if j["model"] == model and j["category"] == category and j["config"] in configs:
                    prompt_data[(j["prompt"], j["config"])].append(
                        j["scores"].get("sarcasm_intensity", 0)
                    )

            for prompt in CATEGORIES[category]:
                values = []
                for config in configs:
                    vals = prompt_data.get((prompt, config), [])
                    values.append(np.mean(vals) if vals else np.nan)

                if not all(np.isnan(values)):
                    ax.plot(x, values, '-', color=COLORS[model], alpha=0.15, linewidth=1)

            # Mean trajectory (bold)
            mean_values = []
            for config in configs:
                all_vals = []
                for prompt in CATEGORIES[category]:
                    vals = prompt_data.get((prompt, config), [])
                    all_vals.extend(vals)
                mean_values.append(np.mean(all_vals) if all_vals else np.nan)

            ax.plot(x, mean_values, 'o-', color=COLORS[model], alpha=1.0,
                    linewidth=3, markersize=10, label=model.upper())

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 10)
        ax.set_xlabel('Layer Position (%)')
        ax.set_ylabel('Sarcasm Intensity' if col_idx == 0 else '')
        ax.set_title(f'{category.upper()}')
        ax.legend(loc='upper right')

        # Add annotation for crossing
        if category != "instruction":
            ax.annotate('', xy=(50, 5), xytext=(20, 5),
                       arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))

    plt.suptitle('Model Comparison: Llama (early peak) vs Gemma (middle peak)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('experiments/exp_001_phase1_replication/figs/fig7_model_comparison.png',
                dpi=150, bbox_inches='tight')
    print("Saved: figs/fig7_model_comparison.png")
    plt.show()


def main():
    judgments = load_judgments()
    print(f"Loaded {len(judgments)} judgments\n")

    plot_trajectories_by_category(judgments)
    plot_subcriteria_by_category(judgments)
    plot_combined_model_comparison(judgments)

    print("\nAll figures saved to figs/")


if __name__ == "__main__":
    main()
