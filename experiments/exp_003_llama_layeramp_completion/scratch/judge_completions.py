#!/usr/bin/env python3
"""
Judge exp_003 completions using claude -p pipeline.

Reads samples from logs/by_prompt, runs parallel claude -p judging,
writes individual JSON judgments.

Usage:
    uv run python experiments/exp_003_llama_layeramp_completion/scratch/judge_completions.py
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

MAX_CONCURRENT = 30
TIMEOUT_S = 60
MAX_RETRIES = 5

LOGS_DIR = Path("experiments/exp_003_llama_layeramp_completion/logs/by_prompt")
JUDGMENTS_DIR = Path("experiments/exp_003_llama_layeramp_completion/judgments")
CRITERIA_FILE = "experiments/exp_001_phase1_replication/rejudge_qwen/judging/criteria.md"
SCHEMA_FILE = "experiments/exp_001_phase1_replication/rejudge_qwen/judging/schema.json"
MODEL = "llama31_8B_Instruct"

ENV = {
    "PATH": subprocess.check_output(["bash", "-c", "echo $PATH"], text=True).strip(),
    "HOME": str(Path.home()),
}


def load_samples() -> list[dict]:
    """Load all samples from logs/by_prompt structure."""
    samples = []
    for prompt_dir in sorted(LOGS_DIR.iterdir()):
        if not prompt_dir.is_dir():
            continue
        for config_dir in sorted(prompt_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            model_dir = config_dir / MODEL
            if not model_dir.exists():
                continue
            for yaml_file in sorted(model_dir.glob("*.yaml")):
                if "debug" in yaml_file.name:
                    continue
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                if data and data.get("completions"):
                    sample_id = f"{prompt_dir.name}__{config_dir.name}"
                    samples.append({
                        "id": sample_id,
                        "prompt_name": prompt_dir.name,
                        "config_name": config_dir.name,
                        "prompt": data.get("prompt", ""),
                        "response": data["completions"][0],
                        "file": str(yaml_file),
                    })
    return samples


def judge_one(sample: dict, schema_str: str) -> tuple[str, dict | None]:
    """Judge a single sample with timeout and exponential backoff retry."""
    # Build sample text for the judge (blind to experimental conditions)
    text = f"PROMPT: {sample['prompt']}\n\nRESPONSE:\n{sample['response']}"

    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                [
                    "claude", "-p", "--model", "haiku",
                    "--setting-sources", "local",
                    "--no-session-persistence",
                    "--tools", "",
                    "--strict-mcp-config",
                    "--system-prompt-file", CRITERIA_FILE,
                    "--output-format", "json",
                    "--json-schema", schema_str,
                ],
                input=text,
                capture_output=True, text=True,
                timeout=TIMEOUT_S,
                env=ENV,
            )
            if result.returncode != 0:
                raise RuntimeError(f"exit {result.returncode}: {result.stderr[:200]}")
            envelope = json.loads(result.stdout)
            return sample["id"], envelope.get("structured_output", envelope)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"  RETRY {attempt+1}/{MAX_RETRIES} ({e.__class__.__name__}) {sample['id']}, waiting {wait}s")
            time.sleep(wait)
    print(f"  FAIL (all retries exhausted): {sample['id']}")
    return sample["id"], None


def main():
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    schema_str = Path(SCHEMA_FILE).read_text()

    print("Loading samples...")
    samples = load_samples()
    print(f"Found {len(samples)} samples")

    # Skip already-judged
    done = {p.stem for p in JUDGMENTS_DIR.glob("*.json")}
    remaining = [s for s in samples if s["id"] not in done]
    print(f"Already judged: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to judge!")
        return

    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {
            pool.submit(judge_one, s, schema_str): s
            for s in remaining
        }
        for future in as_completed(futures):
            sample_id, judgment = future.result()
            sample = futures[future]
            if judgment:
                out = JUDGMENTS_DIR / f"{sample_id}.json"
                # Include metadata alongside judgment
                full = {
                    "sample_id": sample_id,
                    "prompt_name": sample["prompt_name"],
                    "config_name": sample["config_name"],
                    "prompt": sample["prompt"],
                    "response": sample["response"],
                    "judgment": judgment,
                }
                out.write_text(json.dumps(full, indent=2))
                scores = judgment.get("scores", {})
                print(f"OK  {sample_id}: sarc={scores.get('sarcasm_intensity', '?')} wit={scores.get('wit_playfulness', '?')}")
                ok += 1
            else:
                fail += 1

    print(f"\nDone: {ok} ok, {fail} failed, {len(done)} skipped (already done)")
    print(f"Judgments in: {JUDGMENTS_DIR}")


if __name__ == "__main__":
    main()
