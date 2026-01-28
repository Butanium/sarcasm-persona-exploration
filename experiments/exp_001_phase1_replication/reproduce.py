#!/usr/bin/env python3
"""Reproduce key results from exp_001_phase1_replication."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


def main():
    """Run the Phase 1 experiment."""
    cmd = [
        "uv", "run", "python", "tools/run_experiment.py",
        "--prompts", "prompts/phase1",
        "--configs", "configs/sweep_20",
        "--model", "llama31_8B_Instruct",
        "--model-id", "meta-llama/Llama-3.1-8B-Instruct",
        "--url", "http://localhost:8005",  # Note: port may need adjustment
        "--include-base",
        "--max-tokens", "300",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
