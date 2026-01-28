#!/usr/bin/env python3
"""Distribute samples across judging batches."""

import yaml
from pathlib import Path

LOGS_DIR = Path("logs/by_prompt")
JUDGING_DIR = Path("experiments/exp_001_phase1_replication/judging")
SAMPLES_PER_BATCH = 10


def load_samples(model: str) -> list[dict]:
    """Load all samples for a model."""
    samples = []
    for prompt_dir in sorted(LOGS_DIR.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_name = prompt_dir.name.split("_")[0]

        for config_dir in sorted(prompt_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            config_name = config_dir.name

            model_dir = config_dir / model
            if not model_dir.exists():
                continue

            for yaml_file in sorted(model_dir.glob("*.yaml")):
                if "debug" in yaml_file.name:
                    continue
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data and "completions" in data and data["completions"]:
                    samples.append({
                        "id": f"{model}_{prompt_name}_{config_name}",
                        "model": model,
                        "prompt_name": prompt_name,
                        "config_name": config_name,
                        "prompt": data.get("prompt", ""),
                        "response": data["completions"][0],
                    })
    return samples


def write_sample(sample: dict, path: Path):
    """Write sample as text file for judging."""
    content = f"""# Sample: {sample['id']}

## Metadata
- Model: {sample['model']}
- Prompt: {sample['prompt_name']}
- Config: {sample['config_name']}

## User Prompt
{sample['prompt']}

## Response
{sample['response']}
"""
    path.write_text(content)


def main():
    # Load samples from both models
    llama_samples = load_samples("llama31_8B_Instruct")
    gemma_samples = load_samples("gemma3_4B_it")

    print(f"Llama samples: {len(llama_samples)}")
    print(f"Gemma samples: {len(gemma_samples)}")

    all_samples = llama_samples + gemma_samples
    print(f"Total samples: {len(all_samples)}")

    # Distribute to batches
    batches = sorted(JUDGING_DIR.glob("batch_*"))
    print(f"Batches: {len(batches)}")

    for i, sample in enumerate(all_samples):
        batch_idx = i // SAMPLES_PER_BATCH
        if batch_idx >= len(batches):
            print(f"Warning: Not enough batches for sample {i}")
            break

        batch_dir = batches[batch_idx]
        sample_path = batch_dir / "samples" / f"{sample['id']}.txt"
        write_sample(sample, sample_path)

    # Print summary
    for batch in batches:
        n = len(list((batch / "samples").glob("*.txt")))
        if n > 0:
            print(f"{batch.name}: {n} samples")


if __name__ == "__main__":
    main()
