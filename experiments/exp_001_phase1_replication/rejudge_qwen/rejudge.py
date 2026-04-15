#!/usr/bin/env python3
"""Re-judge ALL phase 1 samples (Llama, Gemma, Qwen) using the claude -p pipeline.

One claude -p call per sample, blind to experimental condition.
Samples are collected from all judging batches (batch_001 through batch_015).
Llama/Gemma samples have metadata headers that are stripped to keep the judge blind.
"""

import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

JUDGING_DIR = Path(__file__).parent.parent / "judging"
JUDGMENTS_DIR = Path(__file__).parent / "judgments"
CRITERIA = Path(__file__).parent / "judging" / "criteria.md"
SCHEMA = Path(__file__).parent / "judging" / "schema.json"
MAX_CONCURRENT = 30
TIMEOUT_S = 60
MAX_RETRIES = 5

# Resolve env once, not per subprocess
_ENV = {
    "PATH": subprocess.check_output(["bash", "-c", "echo $PATH"], text=True).strip(),
    "HOME": str(Path.home()),
}


def extract_response_text(raw: str) -> str:
    """Extract only the response text, stripping metadata headers if present.

    Llama/Gemma samples (batch_001-014) have format:
        # Sample: ...
        ## Metadata
        - Model: ...
        - Config: ...   <-- leaks experimental condition!
        ## User Prompt
        ...
        ## Response
        <actual text>

    Qwen samples (batch_015) are raw text with no headers.
    """
    match = re.search(r"## Response\s*\n(.*)", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def collect_samples() -> list[Path]:
    """Collect all .txt sample files from all judging batches."""
    samples = []
    for batch_dir in sorted(JUDGING_DIR.glob("batch_*")):
        samples_dir = batch_dir / "samples"
        if samples_dir.exists():
            samples.extend(sorted(samples_dir.glob("*.txt")))
    return samples


def judge_sample(sample_path: Path, schema_str: str) -> tuple[Path, dict | None]:
    """Judge a single sample via claude -p with timeout and exponential backoff retry."""
    raw = sample_path.read_text()
    text = extract_response_text(raw)
    if not text:
        return sample_path, None

    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                [
                    "claude", "-p",
                    "--model", "haiku",
                    "--setting-sources", "local",
                    "--no-session-persistence",
                    "--tools", "",
                    "--strict-mcp-config",
                    "--system-prompt-file", str(CRITERIA),
                    "--output-format", "json",
                    "--json-schema", schema_str,
                ],
                input=text,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                env=_ENV,
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


def run_parallel(samples: list[Path], schema_str: str):
    """Judge samples in parallel, saving results as they complete."""
    ok, fail = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {
            pool.submit(judge_sample, s, schema_str): s for s in samples
        }
        for future in as_completed(futures):
            sample_path, judgment = future.result()
            if judgment is None:
                fail += 1
                continue
            out_path = JUDGMENTS_DIR / f"{sample_path.stem}.json"
            out_path.write_text(json.dumps(judgment, indent=2))
            sarc = judgment["scores"]["sarcasm_intensity"]
            wit = judgment["scores"]["wit_playfulness"]
            tone = judgment["qualitative"]["dominant_tone"]
            print(f"  OK  sarc={sarc} wit={wit} tone={tone:8s} {sample_path.name}")
            ok += 1
    print(f"\nResults: {ok} ok, {fail} failed")


def run_audit(samples: list[Path], schema_str: str):
    """Judge a small sample sequentially with verbose output for review."""
    # Pick 2 from each model
    by_prefix: dict[str, list[Path]] = {"llama": [], "gemma": [], "qwen": []}
    for s in samples:
        for prefix in by_prefix:
            if prefix in s.name.lower():
                by_prefix[prefix].append(s)
                break
    audit_samples = []
    for ss in by_prefix.values():
        audit_samples.extend(ss[:2])

    print(f"AUDIT MODE: judging {len(audit_samples)} samples (2 per model)\n")
    for i, sample in enumerate(audit_samples):
        print(f"[{i+1}/{len(audit_samples)}] {sample.name}")
        _, judgment = judge_sample(sample, schema_str)
        if judgment is None:
            continue
        out_path = JUDGMENTS_DIR / f"{sample.stem}.json"
        out_path.write_text(json.dumps(judgment, indent=2))
        print(f"  scores:  {judgment['scores']}")
        print(f"  tone:    {judgment['qualitative']['dominant_tone']}")
        print(f"  summary: {judgment['qualitative']['summary']}")
        print()

    print("Audit complete. Review scores above, then run without --audit for full batch.")


def main():
    audit_mode = "--audit" in sys.argv

    samples = collect_samples()
    if not samples:
        print(f"No samples found in {JUDGING_DIR}")
        return

    print(f"Found {len(samples)} total samples across all batches")
    JUDGMENTS_DIR.mkdir(exist_ok=True)
    schema_str = SCHEMA.read_text()

    if audit_mode:
        run_audit(samples, schema_str)
    else:
        done = {p.stem for p in JUDGMENTS_DIR.glob("*.json")}
        remaining = [s for s in samples if s.stem not in done]
        print(f"Judging {len(remaining)} samples ({len(done)} already done, {MAX_CONCURRENT}-wide parallel)")
        run_parallel(remaining, schema_str)


if __name__ == "__main__":
    main()
