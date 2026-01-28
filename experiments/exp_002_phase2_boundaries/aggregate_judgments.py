"""Aggregate Phase 2 judgments and create analysis."""

import yaml
import re
from pathlib import Path
from collections import defaultdict
import json


def parse_judgment_file(path: Path) -> dict:
    """Parse a judgment YAML file and extract metadata from filename."""
    with open(path) as f:
        judgment = yaml.safe_load(f)

    # Filename format: exp2X_model_...config_name.yaml
    name = path.stem

    # Extract experiment type
    exp_match = re.match(r'(exp2[a-z])_', name)
    exp_type = exp_match.group(1) if exp_match else "unknown"

    # Extract config name (everything after last underscore pattern)
    # e.g., "sarcasm_layers_0_10" or "sarcasm_full" or "sarcasm_amp_2x"
    config_match = re.search(r'(sarcasm_[a-z0-9_]+)$', name)
    config_name = config_match.group(1) if config_match else name

    # Try to extract model from experiment type
    model = "unknown"
    if "llama" in name.lower():
        model = "llama"
    elif "gemma" in name.lower():
        model = "gemma"
    elif "qwen" in name.lower():
        model = "qwen"

    # Extract prompt name (between exp type and config)
    parts = name.split('_')
    # Find the prompt portion - it's the middle section
    if exp_match:
        remaining = name[len(exp_match.group(0)):]
        # Remove model prefix if present
        for m in ['llama_fine_', 'gemma_fine_', 'qwen_combo_', 'prompt_llama_', 'prompt_gemma_', 'prompt_qwen_', 'amplify_llama_', 'amplify_gemma_', 'amplify_qwen_', 'layer_amp_']:
            if remaining.startswith(m):
                remaining = remaining[len(m):]
                break
        # Now find where the config starts
        if config_match:
            prompt_end = name.rfind(config_match.group(0))
            prompt_start = name.find(remaining)
            if prompt_start != -1 and prompt_end != -1:
                prompt = name[prompt_start:prompt_end].rstrip('_')
            else:
                prompt = remaining
        else:
            prompt = remaining
    else:
        prompt = "unknown"

    return {
        "exp_type": exp_type,
        "model": model,
        "config": config_name,
        "prompt": prompt,
        "scores": judgment.get("scores", {}),
        "qualitative": judgment.get("qualitative", {}),
        "filename": name,
    }


def aggregate_by_group(judgments: list, group_keys: list) -> dict:
    """Aggregate scores by grouping keys."""
    groups = defaultdict(list)

    for j in judgments:
        key = tuple(j.get(k, "unknown") for k in group_keys)
        groups[key].append(j)

    results = {}
    for key, items in groups.items():
        scores_agg = defaultdict(list)
        for item in items:
            for score_name, score_val in item.get("scores", {}).items():
                if isinstance(score_val, (int, float)):
                    scores_agg[score_name].append(score_val)

        # Calculate mean for each score
        means = {}
        for score_name, values in scores_agg.items():
            if values:
                means[score_name] = sum(values) / len(values)

        key_dict = dict(zip(group_keys, key))
        results[key] = {
            "group": key_dict,
            "n": len(items),
            "mean_scores": means,
        }

    return results


def main():
    judging_dir = Path(__file__).parent / "judging"

    # Collect all judgments
    all_judgments = []
    for batch_dir in sorted(judging_dir.glob("batch_*")):
        for judgment_file in (batch_dir / "judgments").glob("*.yaml"):
            try:
                j = parse_judgment_file(judgment_file)
                all_judgments.append(j)
            except Exception as e:
                print(f"Error parsing {judgment_file}: {e}")

    print(f"Parsed {len(all_judgments)} judgments")

    # Experiment type summary
    exp_counts = defaultdict(int)
    for j in all_judgments:
        exp_counts[j["exp_type"]] += 1
    print("\nExperiments by type:")
    for exp, count in sorted(exp_counts.items()):
        print(f"  {exp}: {count}")

    # 1. Fine-grained layer analysis (exp2a, exp2b)
    print("\n" + "="*60)
    print("FINE-GRAINED LAYER ANALYSIS")
    print("="*60)

    layer_results = aggregate_by_group(
        [j for j in all_judgments if j["exp_type"] in ("exp2a", "exp2b")],
        ["model", "config"]
    )

    # Sort by model then config
    for key in sorted(layer_results.keys()):
        r = layer_results[key]
        model, config = key
        sarcasm = r["mean_scores"].get("sarcasm_intensity", 0)
        wit = r["mean_scores"].get("wit_playfulness", 0)
        cynicism = r["mean_scores"].get("cynicism_negativity", 0)
        print(f"{model:6} {config:30} n={r['n']:2}  sarc={sarcasm:.1f} wit={wit:.1f} cyn={cynicism:.1f}")

    # 2. Qwen layer combinations (exp2c)
    print("\n" + "="*60)
    print("QWEN LAYER COMBINATIONS")
    print("="*60)

    qwen_results = aggregate_by_group(
        [j for j in all_judgments if j["exp_type"] == "exp2c"],
        ["config"]
    )

    for key in sorted(qwen_results.keys()):
        r = qwen_results[key]
        config = key[0]
        sarcasm = r["mean_scores"].get("sarcasm_intensity", 0)
        wit = r["mean_scores"].get("wit_playfulness", 0)
        cynicism = r["mean_scores"].get("cynicism_negativity", 0)
        print(f"{config:35} n={r['n']:2}  sarc={sarcasm:.1f} wit={wit:.1f} cyn={cynicism:.1f}")

    # 3. Prompt boundaries (exp2d)
    print("\n" + "="*60)
    print("PROMPT BOUNDARY ANALYSIS")
    print("="*60)

    prompt_results = aggregate_by_group(
        [j for j in all_judgments if j["exp_type"] == "exp2d"],
        ["model", "prompt"]
    )

    for key in sorted(prompt_results.keys()):
        r = prompt_results[key]
        model, prompt = key
        sarcasm = r["mean_scores"].get("sarcasm_intensity", 0)
        wit = r["mean_scores"].get("wit_playfulness", 0)
        print(f"{model:6} {prompt[:40]:40} n={r['n']:2}  sarc={sarcasm:.1f} wit={wit:.1f}")

    # 4. Amplification strength (exp2e)
    print("\n" + "="*60)
    print("AMPLIFICATION STRENGTH")
    print("="*60)

    amp_results = aggregate_by_group(
        [j for j in all_judgments if j["exp_type"] == "exp2e"],
        ["model", "config"]
    )

    for key in sorted(amp_results.keys()):
        r = amp_results[key]
        model, config = key
        sarcasm = r["mean_scores"].get("sarcasm_intensity", 0)
        wit = r["mean_scores"].get("wit_playfulness", 0)
        cynicism = r["mean_scores"].get("cynicism_negativity", 0)
        exagg = r["mean_scores"].get("exaggeration_stakes", 0)
        print(f"{model:6} {config:25} n={r['n']:2}  sarc={sarcasm:.1f} wit={wit:.1f} cyn={cynicism:.1f} exag={exagg:.1f}")

    # 5. Layer + Amplification combos (exp2g)
    print("\n" + "="*60)
    print("LAYER + AMPLIFICATION COMBINATIONS")
    print("="*60)

    combo_results = aggregate_by_group(
        [j for j in all_judgments if j["exp_type"] == "exp2g"],
        ["config"]
    )

    for key in sorted(combo_results.keys()):
        r = combo_results[key]
        config = key[0]
        sarcasm = r["mean_scores"].get("sarcasm_intensity", 0)
        wit = r["mean_scores"].get("wit_playfulness", 0)
        cynicism = r["mean_scores"].get("cynicism_negativity", 0)
        exagg = r["mean_scores"].get("exaggeration_stakes", 0)
        print(f"{config:35} n={r['n']:2}  sarc={sarcasm:.1f} wit={wit:.1f} cyn={cynicism:.1f} exag={exagg:.1f}")

    # Save aggregated results as JSON
    output = {
        "total_samples": len(all_judgments),
        "exp_counts": dict(exp_counts),
        "fine_grained_layers": {str(k): v for k, v in layer_results.items()},
        "qwen_combos": {str(k): v for k, v in qwen_results.items()},
        "prompt_boundaries": {str(k): v for k, v in prompt_results.items()},
        "amplification": {str(k): v for k, v in amp_results.items()},
        "layer_amp_combos": {str(k): v for k, v in combo_results.items()},
    }

    output_path = Path(__file__).parent / "aggregated_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved aggregated results to {output_path}")


if __name__ == "__main__":
    main()
