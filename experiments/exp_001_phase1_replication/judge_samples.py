#!/usr/bin/env python3
"""
Run LLM-as-judge evaluation on Phase 1 samples.
Uses Haiku for cost-effective batch judging.
"""

import yaml
import json
import os
from pathlib import Path
from anthropic import Anthropic

CRITERIA = """
Rate each dimension 0-10:

### sarcasm_intensity
0 = Completely sincere | 5 = Moderate sarcasm | 10 = Extreme sarcasm

### wit_playfulness
0 = Dry, formal | 5 = Consistently clever | 10 = Brilliantly witty

### cynicism_negativity
0 = Optimistic | 5 = Mildly cynical | 10 = Deeply pessimistic

### exaggeration_stakes
0 = Proportionate | 5 = Noticeable exaggeration | 10 = Absurdly over-the-top

### meta_awareness
0 = No meta-commentary | 5 = Regular meta-references | 10 = Predominantly meta

Also provide:
- dominant_tone: one of [sincere, playful, sarcastic, cynical, absurdist, neutral]
- notable_features: 1-3 distinctive stylistic elements
"""

JUDGE_PROMPT = """You are evaluating an AI assistant's response for stylistic qualities.

PROMPT: {prompt}

RESPONSE:
{response}

{criteria}

Respond ONLY with valid YAML:
```yaml
scores:
  sarcasm_intensity: <0-10>
  wit_playfulness: <0-10>
  cynicism_negativity: <0-10>
  exaggeration_stakes: <0-10>
  meta_awareness: <0-10>
qualitative:
  dominant_tone: "<tone>"
  notable_features:
    - "feature 1"
```
"""


def load_samples(logs_dir: Path, model: str) -> list[dict]:
    """Load all samples for a model from by_prompt logs."""
    samples = []
    for prompt_dir in logs_dir.glob("*"):
        if not prompt_dir.is_dir():
            continue
        prompt_name = prompt_dir.name.split("_")[0]

        for config_dir in prompt_dir.iterdir():
            if not config_dir.is_dir():
                continue
            config_name = config_dir.name

            model_dir = config_dir / model
            if not model_dir.exists():
                continue

            for yaml_file in model_dir.glob("*.yaml"):
                if "debug" in yaml_file.name:
                    continue
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data and "completions" in data:
                    samples.append({
                        "prompt_name": prompt_name,
                        "config_name": config_name,
                        "model": model,
                        "prompt": data.get("prompt", ""),
                        "response": data["completions"][0] if data["completions"] else "",
                        "file": str(yaml_file),
                    })
    return samples


def judge_sample(client: Anthropic, sample: dict) -> dict:
    """Run a single judgment."""
    prompt = JUDGE_PROMPT.format(
        prompt=sample["prompt"],
        response=sample["response"],
        criteria=CRITERIA,
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    # Extract YAML from response
    if "```yaml" in text:
        yaml_text = text.split("```yaml")[1].split("```")[0]
    elif "```" in text:
        yaml_text = text.split("```")[1].split("```")[0]
    else:
        yaml_text = text

    try:
        judgment = yaml.safe_load(yaml_text)
    except Exception as e:
        judgment = {"error": str(e), "raw": text}

    return {
        **sample,
        "judgment": judgment,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama31_8B_Instruct")
    parser.add_argument("--logs-dir", default="logs/by_prompt")
    parser.add_argument("--output", default="experiments/exp_001_phase1_replication/judgments/results.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    client = Anthropic()
    logs_dir = Path(args.logs_dir)

    print(f"Loading samples for {args.model}...")
    samples = load_samples(logs_dir, args.model)
    print(f"Found {len(samples)} samples")

    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    results = []
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] {sample['prompt_name']} × {sample['config_name']}...", end=" ", flush=True)
        try:
            result = judge_sample(client, sample)
            results.append(result)
            if "error" not in result.get("judgment", {}):
                scores = result["judgment"].get("scores", {})
                print(f"sarcasm={scores.get('sarcasm_intensity', '?')}")
            else:
                print("ERROR")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({**sample, "judgment": {"error": str(e)}})

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
