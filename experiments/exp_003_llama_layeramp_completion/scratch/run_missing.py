#!/usr/bin/env python3
"""
Run missing Llama layer-amp and full-sarcasm experiments.

Generates completions for:
1. Layer-amp 0-20% @ 2x/3x — missing 5 prompts each
2. Layer-amp 40-60% @ 2x/3x — missing 6 prompts each
3. Full sarcasm @ 2x/3x — all 9 prompts each

Usage:
    uv run python experiments/exp_003_llama_layeramp_completion/scratch/run_missing.py \
        --url http://localhost:8000
"""

import argparse
import sys
from pathlib import Path

# Add tools/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from run_experiment import (
    load_prompt,
    load_config,
    run_single_experiment,
)

MODEL_NAME = "llama31_8B_Instruct"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PROMPTS_DIR = Path("prompts/phase1")
LOGS_DIR = Path("experiments/exp_003_llama_layeramp_completion/logs")

# Existing Llama layer-amp data (already collected)
EXISTING = {
    "sarcasm_layers_0_20_2x": {
        "direct-how-are-you", "direct-mondays",
        "instruction-exercise-reasons", "instruction-movie-summary",
    },
    "sarcasm_layers_0_20_3x": {
        "direct-how-are-you", "direct-mondays",
        "instruction-exercise-reasons", "instruction-movie-summary",
    },
    "sarcasm_layers_40_60_2x": {
        "direct-how-are-you", "direct-mondays",
        "instruction-exercise-reasons",
    },
    "sarcasm_layers_40_60_3x": {
        "direct-how-are-you", "direct-mondays",
        "instruction-exercise-reasons",
    },
    # Full sarcasm — none exist for Llama yet
    "sarcasm_full_2x": set(),
    "sarcasm_full_3x": set(),
}

ALL_PROMPTS = {
    "creative-morning-routine", "creative-pineapple-pizza", "creative-reddit",
    "direct-first-job-advice", "direct-how-are-you", "direct-mondays",
    "instruction-exercise-reasons", "instruction-movie-summary", "instruction-photosynthesis",
}

CONFIG_FILES = {
    "sarcasm_layers_0_20_2x": "configs/sweep_layer_amplify/llama_0_20_2x.yaml",
    "sarcasm_layers_0_20_3x": "configs/sweep_layer_amplify/llama_0_20_3x.yaml",
    "sarcasm_layers_40_60_2x": "configs/sweep_layer_amplify/llama_40_60_2x.yaml",
    "sarcasm_layers_40_60_3x": "configs/sweep_layer_amplify/llama_40_60_3x.yaml",
    "sarcasm_full_2x": "configs/sweep_amplify/full_2x.yaml",
    "sarcasm_full_3x": "configs/sweep_amplify/full_3x.yaml",
}

PROMPT_FILES = {
    "creative-morning-routine": "prompts/phase1/creative_morning.yaml",
    "creative-pineapple-pizza": "prompts/phase1/creative_pineapple.yaml",
    "creative-reddit": "prompts/phase1/creative_reddit.yaml",
    "direct-first-job-advice": "prompts/phase1/direct_first_job.yaml",
    "direct-how-are-you": "prompts/phase1/direct_howru.yaml",
    "direct-mondays": "prompts/phase1/direct_mondays.yaml",
    "instruction-exercise-reasons": "prompts/phase1/instruction_exercise.yaml",
    "instruction-movie-summary": "prompts/phase1/instruction_movie.yaml",
    "instruction-photosynthesis": "prompts/phase1/instruction_photosynthesis.yaml",
}


def main():
    parser = argparse.ArgumentParser(description="Run missing Llama layer-amp experiments")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be run without executing")
    args = parser.parse_args()

    # Build list of (config_name, prompt_name) pairs to run
    to_run = []
    for config_name, existing_prompts in EXISTING.items():
        missing = ALL_PROMPTS - existing_prompts
        for prompt_name in sorted(missing):
            to_run.append((config_name, prompt_name))

    print(f"Total experiments to run: {len(to_run)}")
    print()

    if args.dry_run:
        for config_name, prompt_name in to_run:
            print(f"  {prompt_name} × {config_name}")
        return

    results = []
    for i, (config_name, prompt_name) in enumerate(to_run, 1):
        config_path = Path(CONFIG_FILES[config_name])
        prompt_path = Path(PROMPT_FILES[prompt_name])

        config_data = load_config(config_path)
        prompt_data = load_prompt(prompt_path)

        print(f"[{i}/{len(to_run)}] {prompt_name} × {config_name}...", end=" ", flush=True)

        result = run_single_experiment(
            base_url=args.url,
            model_name=MODEL_NAME,
            model_id=MODEL_ID,
            prompt_path=prompt_path,
            prompt_data=prompt_data,
            config_path=config_path,
            config_data=config_data,
            request_id="exp003_llama_layeramp",
            logs_dir=LOGS_DIR,
            max_tokens=200,
            temperature=0.7,
            n=1,
        )
        results.append(result)

        if result.get("completions"):
            preview = result["completions"][0][:80].replace("\n", " ")
            print(f"'{preview}...'")
        else:
            print("(no output)")

    successful = len([r for r in results if "error" not in r])
    print(f"\nDone: {successful}/{len(to_run)} successful")
    print(f"Logs written to: {LOGS_DIR}")


if __name__ == "__main__":
    main()
