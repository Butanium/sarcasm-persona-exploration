"""Generate per-prompt sarcasm-by-layer-range plot for 3 models."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_parquet("article/data/v2_judgments.parquet")

CONFIG_ORDER = [
    "base",
    "sarcasm_layers_0_20",
    "sarcasm_layers_20_40",
    "sarcasm_layers_40_60",
    "sarcasm_layers_60_80",
    "sarcasm_layers_80_100",
    "sarcasm_full",
]
LAYER_LABELS = ["base", "0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "full"]

MODEL_ORDER = ["llama", "gemma", "qwen"]
MODEL_TITLES = {
    "llama": "Llama 3.1 8B",
    "gemma": "Gemma 3 4B",
    "qwen": "Qwen 2.5 7B",
}

PROMPT_SHORT = {
    "creative-morning-routine": "Morning Routine",
    "creative-pineapple-pizza": "Pineapple Pizza",
    "creative-reddit": "Reddit Post",
    "direct-first-job-advice": "First Job Advice",
    "direct-how-are-you": "How Are You",
    "direct-mondays": "Mondays",
    "instruction-exercise-reasons": "Exercise Reasons",
    "instruction-movie-summary": "Movie Summary",
    "instruction-photosynthesis": "Photosynthesis",
}

# 9 distinct colors: tab10 minus the last one
COLORS = [mpl.colormaps["tab10"](i) for i in range(9)]

prompts_sorted = sorted(df["prompt"].unique())

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

for ax, model in zip(axes, MODEL_ORDER):
    mdf = df[df["model"] == model]
    for i, prompt in enumerate(prompts_sorted):
        pdf = mdf[mdf["prompt"] == prompt].set_index("config").reindex(CONFIG_ORDER)
        ax.plot(
            range(len(CONFIG_ORDER)),
            pdf["sarcasm"].values,
            color=COLORS[i],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=PROMPT_SHORT.get(prompt, prompt),
        )
    ax.set_xticks(range(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_title(MODEL_TITLES[model], fontsize=11, fontweight="bold")
    ax.set_ylim(-0.5, 10.5)
    ax.set_ylabel("Sarcasm score" if model == MODEL_ORDER[0] else "")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

# Single legend below the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    fontsize=8,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08),
)

fig.suptitle("Per-Prompt Sarcasm Across Layer Ranges", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(
    "article/figures/export/per_prompt_consistency.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
print("Saved article/figures/export/per_prompt_consistency.png")
