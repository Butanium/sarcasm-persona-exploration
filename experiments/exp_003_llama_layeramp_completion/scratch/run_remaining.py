#!/usr/bin/env python3
"""
Run remaining 8 sarcasm_full_3x experiments that failed due to server shutdown.

Usage:
    uv run python experiments/exp_003_llama_layeramp_completion/scratch/run_remaining.py \
        --url http://localhost:8010
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from run_experiment import load_prompt, load_config, run_single_experiment

MODEL_NAME = "llama31_8B_Instruct"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LOGS_DIR = Path("experiments/exp_003_llama_layeramp_completion/logs")
CONFIG_PATH = Path("configs/sweep_amplify/full_3x.yaml")

REMAINING_PROMPTS = [
    ("creative-pineapple-pizza", "prompts/phase1/creative_pineapple.yaml"),
    ("creative-reddit", "prompts/phase1/creative_reddit.yaml"),
    ("direct-first-job-advice", "prompts/phase1/direct_first_job.yaml"),
    ("direct-how-are-you", "prompts/phase1/direct_howru.yaml"),
    ("direct-mondays", "prompts/phase1/direct_mondays.yaml"),
    ("instruction-exercise-reasons", "prompts/phase1/instruction_exercise.yaml"),
    ("instruction-movie-summary", "prompts/phase1/instruction_movie.yaml"),
    ("instruction-photosynthesis", "prompts/phase1/instruction_photosynthesis.yaml"),
]


def main():
    parser = argparse.ArgumentParser(description="Run remaining sarcasm_full_3x experiments")
    parser.add_argument("--url", default="http://localhost:8010", help="vLLM server URL")
    args = parser.parse_args()

    config_data = load_config(CONFIG_PATH)

    for i, (prompt_name, prompt_file) in enumerate(REMAINING_PROMPTS, 1):
        prompt_path = Path(prompt_file)
        prompt_data = load_prompt(prompt_path)

        print(f"[{i}/{len(REMAINING_PROMPTS)}] {prompt_name} × sarcasm_full_3x...", end=" ", flush=True)

        result = run_single_experiment(
            base_url=args.url,
            model_name=MODEL_NAME,
            model_id=MODEL_ID,
            prompt_path=prompt_path,
            prompt_data=prompt_data,
            config_path=CONFIG_PATH,
            config_data=config_data,
            request_id="exp003_llama_layeramp",
            logs_dir=LOGS_DIR,
            max_tokens=200,
            temperature=0.7,
            n=1,
        )

        if result.get("completions"):
            preview = result["completions"][0][:80].replace("\n", " ")
            print(f"'{preview}...'")
        else:
            print("(no output)")

    print(f"\nDone! All 8 remaining experiments completed.")
    print(f"Logs written to: {LOGS_DIR}")


if __name__ == "__main__":
    main()
