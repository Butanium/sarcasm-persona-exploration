#!/usr/bin/env python3
"""
Prepare best-slice boundary samples for judging and run the judging pipeline.

Extracts completions from the best 20% layer slice per model:
  - Llama: layers_0_20
  - Gemma: layers_40_60
  - Qwen: layers_20_40

Then runs claude -p judging on each sample in parallel.
"""

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

BEST_SLICES = {
    "llama31_8B_Instruct": "sarcasm_layers_0_20",
    "gemma3_4B_it": "sarcasm_layers_40_60",
    "qwen25_7B_Instruct": "sarcasm_layers_20_40",
}

LOGS_BASE = Path("logs/by_request")
REQUEST_IDS = {
    "llama31_8B_Instruct": "exp_bestslice_llama",
    "gemma3_4B_it": "exp_bestslice_gemma",
    "qwen25_7B_Instruct": "exp_bestslice_qwen",
}

EXP_DIR = Path("experiments/exp_003_bestslice_boundaries")
SAMPLES_DIR = EXP_DIR / "samples"
JUDGMENTS_DIR = EXP_DIR / "judgments"
CRITERIA_FILE = Path("experiments/exp_001_phase1_replication/rejudge_qwen/judging/criteria.md")
SCHEMA_FILE = Path("experiments/exp_001_phase1_replication/rejudge_qwen/judging/schema.json")

MAX_CONCURRENT = 30
TIMEOUT_S = 60
MAX_RETRIES = 5

ENV = {
    "PATH": subprocess.check_output(["bash", "-c", "echo $PATH"], text=True).strip(),
    "HOME": str(Path.home()),
}


def extract_samples():
    """Extract best-slice completions into sample text files."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for model, config_name in BEST_SLICES.items():
        request_id = REQUEST_IDS[model]
        request_dir = LOGS_BASE / request_id
        if not request_dir.exists():
            print(f"Skipping {model}: {request_dir} not found yet")
            continue

        for prompt_dir in sorted(request_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            prompt_name = prompt_dir.name.rsplit("_", 1)[0]  # strip hash
            log_file = prompt_dir / f"{config_name}.yaml"
            assert log_file.exists(), f"Missing log: {log_file}"

            with open(log_file) as f:
                data = yaml.safe_load(f)

            completion = data["completions"][0]
            sample_name = f"{model}__{prompt_name}__{config_name}"
            sample_file = SAMPLES_DIR / f"{sample_name}.txt"
            sample_file.write_text(completion)
            count += 1

    print(f"Extracted {count} samples to {SAMPLES_DIR}")
    return count


def judge_one(sample_path: Path) -> tuple[Path, dict | None]:
    """Judge a single sample with timeout and exponential backoff retry."""
    text = sample_path.read_text()
    schema_str = SCHEMA_FILE.read_text()

    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                [
                    "claude", "-p", "--model", "haiku",
                    "--setting-sources", "local",
                    "--no-session-persistence",
                    "--tools", "",
                    "--strict-mcp-config",
                    "--system-prompt-file", str(CRITERIA_FILE),
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
            return sample_path, envelope.get("structured_output", envelope)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as e:
            wait = 2 ** attempt
            print(f"  RETRY {attempt+1}/{MAX_RETRIES} ({e.__class__.__name__}) {sample_path.name}, waiting {wait}s")
            time.sleep(wait)
    print(f"  FAIL (all retries exhausted): {sample_path.name}")
    return sample_path, None


def run_judging():
    """Run judging pipeline on all samples."""
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    samples = sorted(SAMPLES_DIR.glob("*.txt"))
    done = {p.stem for p in JUDGMENTS_DIR.glob("*.json")}
    remaining = [s for s in samples if s.stem not in done]

    print(f"Judging {len(remaining)} samples ({len(done)} already done)")

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(judge_one, s): s for s in remaining}
        ok, fail = 0, 0
        for future in as_completed(futures):
            sample_path, judgment = future.result()
            if judgment:
                out = JUDGMENTS_DIR / f"{sample_path.stem}.json"
                out.write_text(json.dumps(judgment, indent=2))
                scores = judgment.get("scores", {})
                print(f"OK  {sample_path.name}: sarcasm={scores.get('sarcasm_intensity')}, wit={scores.get('wit_playfulness')}")
                ok += 1
            else:
                fail += 1
        print(f"\nDone: {ok} ok, {fail} failed, {len(done)} skipped (already done)")


def aggregate_csv():
    """Aggregate judgments into CSV."""
    output_dir = Path("article/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bestslice_boundaries.csv"

    rows = []
    for jf in sorted(JUDGMENTS_DIR.glob("*.json")):
        parts = jf.stem.split("__")
        assert len(parts) == 3, f"Unexpected filename format: {jf.name}"
        model, prompt_type, _config = parts

        with open(jf) as f:
            judgment = json.load(f)

        scores = judgment["scores"]
        rows.append({
            "model": model,
            "prompt_type": prompt_type,
            "sarcasm": scores["sarcasm_intensity"],
            "wit": scores["wit_playfulness"],
            "cynicism": scores["cynicism_negativity"],
            "exaggeration": scores["exaggeration_stakes"],
            "meta": scores["meta_awareness"],
        })

    # Write CSV
    import csv
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "prompt_type", "sarcasm", "wit", "cynicism", "exaggeration", "meta"])
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["model"], r["prompt_type"])):
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {output_file}")
    return output_file


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: prepare_and_judge.py [extract|judge|aggregate|all]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "extract":
        extract_samples()
    elif cmd == "judge":
        run_judging()
    elif cmd == "aggregate":
        aggregate_csv()
    elif cmd == "all":
        extract_samples()
        run_judging()
        aggregate_csv()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
