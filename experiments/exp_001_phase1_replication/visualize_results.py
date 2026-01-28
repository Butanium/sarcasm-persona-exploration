#!/usr/bin/env python3
"""
Visualization of Phase 1 Results: Sarcasm Layer Decomposition

Key finding: Layer effects are architecture-specific!
- Llama 3.1 8B: Sarcasm concentrated in early layers (0-40%)
- Gemma 3 4B: Sarcasm concentrated in middle layers (40-60%)
"""

# %% Imports and setup
import yaml
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE = (12, 5)
COLORS = {
    'llama': '#2E86AB',  # Blue
    'gemma': '#A23B72',  # Magenta
    'qwen': '#F18F01',   # Orange
}

# %% Load judgment data
JUDGING_DIR = Path("experiments/exp_001_phase1_replication/judging")

def load_judgments():
    """Load all judgment YAML files."""
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
                elif "qwen" in name:
                    model = "qwen"
                else:
                    model = "unknown"

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
                    config = "unknown"

                judgments.append({
                    "file": str(yaml_file),
                    "model": model,
                    "config": config,
                    "scores": data["scores"],
                })
    return judgments

judgments = load_judgments()
print(f"Loaded {len(judgments)} judgments")

# %% Aggregate by model and config
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
                avg_scores[f"{dim}_std"] = np.std(values) if len(values) > 1 else 0
        results[(model, config)] = {
            "n": len(scores_list),
            "avg": avg_scores,
        }
    return results

results = aggregate_by_model_config(judgments)

# %% Plot 1: Sarcasm intensity by layer range (main finding)
fig, ax = plt.subplots(figsize=FIGSIZE)

configs = ["base", "full", "0-20", "20-40", "40-60", "60-80", "80-100"]
x = np.arange(len(configs))
width = 0.35

for i, model in enumerate(["llama", "gemma"]):
    values = []
    stds = []
    for config in configs:
        key = (model, config)
        if key in results:
            values.append(results[key]["avg"].get("sarcasm_intensity", 0))
            stds.append(results[key]["avg"].get("sarcasm_intensity_std", 0))
        else:
            values.append(0)
            stds.append(0)

    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, values, width, label=model.upper(), color=COLORS[model],
                  yerr=stds, capsize=3, alpha=0.85)

ax.set_xlabel('Layer Configuration', fontsize=12)
ax.set_ylabel('Sarcasm Intensity (0-10)', fontsize=12)
ax.set_title('Sarcasm Intensity by Layer Range\n(Architecture-Specific Effects)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.legend(title='Model')
ax.set_ylim(0, 10)

# Add annotations for key finding
ax.annotate('Peak: Early layers', xy=(2, 5.5), xytext=(1, 7.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['llama']),
            fontsize=10, color=COLORS['llama'])
ax.annotate('Peak: Middle layers', xy=(4, 5.6), xytext=(5, 7.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['gemma']),
            fontsize=10, color=COLORS['gemma'])

plt.tight_layout()
plt.savefig('experiments/exp_001_phase1_replication/figs/fig1_sarcasm_by_layer.png', dpi=150, bbox_inches='tight')
plt.savefig('experiments/exp_001_phase1_replication/figs/fig1_sarcasm_by_layer.pdf', bbox_inches='tight')
print("Saved: fig1_sarcasm_by_layer.png")
plt.show()

# %% Plot 2: Multi-dimension heatmap comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

dimensions = ["sarcasm_intensity", "wit_playfulness", "cynicism_negativity",
              "exaggeration_stakes", "meta_awareness"]
dim_labels = ["Sarcasm", "Wit", "Cynicism", "Exaggeration", "Meta"]
configs_layer = ["0-20", "20-40", "40-60", "60-80", "80-100"]

for ax_idx, model in enumerate(["llama", "gemma"]):
    data = np.zeros((len(dimensions), len(configs_layer)))

    for i, dim in enumerate(dimensions):
        for j, config in enumerate(configs_layer):
            key = (model, config)
            if key in results:
                data[i, j] = results[key]["avg"].get(dim, 0)

    im = axes[ax_idx].imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=8)
    axes[ax_idx].set_xticks(range(len(configs_layer)))
    axes[ax_idx].set_xticklabels(configs_layer)
    axes[ax_idx].set_yticks(range(len(dimensions)))
    axes[ax_idx].set_yticklabels(dim_labels)
    axes[ax_idx].set_xlabel('Layer Range (%)')
    axes[ax_idx].set_title(f'{model.upper()} - Score Heatmap', fontweight='bold')

    # Add value annotations
    for i in range(len(dimensions)):
        for j in range(len(configs_layer)):
            text = axes[ax_idx].text(j, i, f'{data[i, j]:.1f}',
                                      ha='center', va='center', fontsize=9,
                                      color='white' if data[i, j] > 4 else 'black')

fig.colorbar(im, ax=axes.ravel().tolist(), label='Score (0-10)', shrink=0.8)
fig.suptitle('All Dimensions by Layer Range', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('experiments/exp_001_phase1_replication/figs/fig2_heatmap_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('experiments/exp_001_phase1_replication/figs/fig2_heatmap_comparison.pdf', bbox_inches='tight')
print("Saved: fig2_heatmap_comparison.png")
plt.show()

# %% Plot 3: Line plot showing layer progression
fig, ax = plt.subplots(figsize=FIGSIZE)

layer_configs = ["0-20", "20-40", "40-60", "60-80", "80-100"]
x = [10, 30, 50, 70, 90]  # Midpoints of each range

for model in ["llama", "gemma"]:
    values = []
    for config in layer_configs:
        key = (model, config)
        if key in results:
            values.append(results[key]["avg"].get("sarcasm_intensity", 0))
        else:
            values.append(0)

    ax.plot(x, values, 'o-', label=model.upper(), color=COLORS[model],
            linewidth=2.5, markersize=10)

# Add reference lines for base and full
for model in ["llama", "gemma"]:
    base_val = results.get((model, "base"), {}).get("avg", {}).get("sarcasm_intensity", 0)
    full_val = results.get((model, "full"), {}).get("avg", {}).get("sarcasm_intensity", 0)
    ax.axhline(y=base_val, color=COLORS[model], linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(y=full_val, color=COLORS[model], linestyle='--', alpha=0.5, linewidth=1.5)

ax.set_xlabel('Layer Position (% of total layers)', fontsize=12)
ax.set_ylabel('Sarcasm Intensity (0-10)', fontsize=12)
ax.set_title('Sarcasm Distribution Across Layers\n(Dashed = Full LoRA, Dotted = Base)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.legend(title='Model')

plt.tight_layout()
plt.savefig('experiments/exp_001_phase1_replication/figs/fig3_layer_progression.png', dpi=150, bbox_inches='tight')
plt.savefig('experiments/exp_001_phase1_replication/figs/fig3_layer_progression.pdf', bbox_inches='tight')
print("Saved: fig3_layer_progression.png")
plt.show()

# %% Summary statistics table
print("\n" + "="*60)
print("SUMMARY: Architecture-Specific Layer Effects")
print("="*60)

for model in ["llama", "gemma"]:
    print(f"\n{model.upper()}:")
    print(f"{'Config':<10} {'N':>4} {'Sarcasm':>10} {'Wit':>10} {'Cynicism':>10}")
    print("-" * 50)
    for config in ["base", "full", "0-20", "20-40", "40-60", "60-80", "80-100"]:
        key = (model, config)
        if key in results:
            r = results[key]
            avg = r["avg"]
            print(f"{config:<10} {r['n']:>4} "
                  f"{avg.get('sarcasm_intensity', 0):>10.1f} "
                  f"{avg.get('wit_playfulness', 0):>10.1f} "
                  f"{avg.get('cynicism_negativity', 0):>10.1f}")

print("\n" + "="*60)
print("KEY FINDING:")
print("- Llama: Peak sarcasm in layers 0-40% (early)")
print("- Gemma: Peak sarcasm in layers 40-60% (middle)")
print("- Same training → Different layer distributions!")
print("="*60)
