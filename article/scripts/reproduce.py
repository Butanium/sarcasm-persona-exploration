#!/usr/bin/env python3
"""Reproduce the data inspection outputs: PNG figures and sample inspection report.

Run with: uv run article/scripts/reproduce.py
"""

import subprocess
import sys


def main():
    print("=== Step 1: Generate PNG figures ===")
    result = subprocess.run(
        [sys.executable, "article/scripts/generate_pngs.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("FAILED:", result.stderr)
        sys.exit(1)

    print("\n=== Step 2: Generate sample inspection report ===")
    result = subprocess.run(
        [sys.executable, "article/scripts/inspect_samples.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("FAILED:", result.stderr)
        sys.exit(1)

    print("\n=== Done! ===")
    print("Figures: article/figures/*.png")
    print("Report: article/data/sample_inspection_report.md")


if __name__ == "__main__":
    main()
