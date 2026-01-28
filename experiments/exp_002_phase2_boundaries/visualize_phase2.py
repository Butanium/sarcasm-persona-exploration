"""Visualize Phase 2 results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    results_path = Path(__file__).parent / "aggregated_results.json"
    with open(results_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Phase 2: Sarcasm Persona Boundaries", fontsize=14)

    # 1. Fine-grained layer analysis
    ax = axes[0, 0]
    ax.set_title("Fine-Grained Layers (10% slices)")

    # Llama data
    llama_layers = ["0-10%", "10-20%", "20-30%", "30-40%"]
    llama_sarc = [0.8, 2.0, 1.0, 3.6]

    # Gemma data
    gemma_layers = ["30-40%", "40-50%", "50-60%", "60-70%"]
    gemma_sarc = [1.4, 2.9, 2.8, 1.8]

    x = np.arange(4)
    width = 0.35
    ax.bar(x - width/2, llama_sarc, width, label="Llama", color="steelblue")
    ax.bar(x + width/2, gemma_sarc, width, label="Gemma", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(["Slice 1", "Slice 2", "Slice 3", "Slice 4"])
    ax.set_ylabel("Sarcasm Score")
    ax.legend()
    ax.text(0.5, -0.15, "Llama: 0-40%  |  Gemma: 30-70%", transform=ax.transAxes, ha="center", fontsize=8)

    # 2. Qwen combinations
    ax = axes[0, 1]
    ax.set_title("Qwen Layer Combinations")
    qwen_configs = ["0-50%", "50-100%", "Bookends", "Middle"]
    qwen_sarc = [5.6, 6.6, 6.6, 7.1]
    colors = ["lightblue", "lightblue", "lightblue", "steelblue"]
    bars = ax.bar(qwen_configs, qwen_sarc, color=colors)
    ax.set_ylabel("Sarcasm Score")
    ax.set_ylim(0, 10)
    ax.axhline(y=7.4, color="red", linestyle="--", alpha=0.5, label="Full LoRA")
    ax.legend()

    # 3. Amplification curves
    ax = axes[0, 2]
    ax.set_title("Amplification Strength Effect")
    amps = [0.5, 1.5, 2.0, 3.0]

    llama_amp_sarc = [5.3, 7.8, 8.2, 9.0]
    gemma_amp_sarc = [4.9, 7.6, 8.1, 7.3]
    qwen_amp_sarc = [3.8, 7.2, 8.2, 8.6]

    ax.plot(amps, llama_amp_sarc, "o-", label="Llama", color="steelblue")
    ax.plot(amps, gemma_amp_sarc, "s-", label="Gemma", color="coral")
    ax.plot(amps, qwen_amp_sarc, "^-", label="Qwen", color="forestgreen")
    ax.set_xlabel("Amplification")
    ax.set_ylabel("Sarcasm Score")
    ax.set_ylim(0, 10)
    ax.legend()

    # 4. Wit collapse at high amplification
    ax = axes[1, 0]
    ax.set_title("Wit vs Sarcasm at High Amplification")

    llama_amp_wit = [6.1, 6.7, 6.4, 6.2]
    gemma_amp_wit = [5.9, 4.7, 3.4, 1.3]
    qwen_amp_wit = [4.8, 6.1, 6.6, 4.4]

    ax.plot(amps, llama_amp_wit, "o-", label="Llama wit", color="steelblue")
    ax.plot(amps, gemma_amp_wit, "s-", label="Gemma wit", color="coral")
    ax.plot(amps, qwen_amp_wit, "^-", label="Qwen wit", color="forestgreen")
    ax.set_xlabel("Amplification")
    ax.set_ylabel("Wit Score")
    ax.set_ylim(0, 10)
    ax.legend()
    ax.annotate("Gemma wit collapses!", xy=(3, 1.3), xytext=(2.5, 3),
                arrowprops=dict(arrowstyle="->", color="red"), color="red")

    # 5. Layer + Amplification combos
    ax = axes[1, 1]
    ax.set_title("Layer Selection vs Amplification")
    configs = ["0-20%\n@ 2x", "0-20%\n@ 3x", "40-60%\n@ 2x", "40-60%\n@ 3x"]
    sarc_values = [3.0, 3.2, 6.9, 8.2]
    colors = ["lightgray", "lightgray", "steelblue", "steelblue"]
    ax.bar(configs, sarc_values, color=colors)
    ax.set_ylabel("Sarcasm Score")
    ax.set_ylim(0, 10)
    ax.axhline(y=3.2, color="red", linestyle=":", alpha=0.5)
    ax.text(0.25, 4, "Wrong layers\n(even at 3x)", ha="center", fontsize=8, color="gray")
    ax.text(0.75, 9, "Right layers\n(strong at 2x+)", ha="center", fontsize=8, color="steelblue",
            transform=ax.transAxes)

    # 6. Prompt robustness
    ax = axes[1, 2]
    ax.set_title("Prompt Type Effects (Full LoRA)")
    prompt_types = ["Formal", "Emotional", "Anti-sarc", "Technical", "Raw", "Prefill"]
    base_sarc = [0, 0.5, 1.0, 0, 2.5, 1.0]
    full_sarc = [8, 8, 8.5, 8, 7, 5.5]

    x = np.arange(len(prompt_types))
    width = 0.35
    ax.bar(x - width/2, base_sarc, width, label="Base", color="lightgray")
    ax.bar(x + width/2, full_sarc, width, label="Full LoRA", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_types, rotation=45, ha="right")
    ax.set_ylabel("Sarcasm Score")
    ax.set_ylim(0, 10)
    ax.legend()

    plt.tight_layout()
    output_path = Path(__file__).parent / "phase2_results.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved to {output_path}")

    # Also save PNG for quick viewing
    png_path = Path(__file__).parent / "phase2_results.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()
