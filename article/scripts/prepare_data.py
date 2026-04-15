#!/usr/bin/env python3
"""Prepare experiment data as clean CSVs for the Quarto report.

Reads:
  - experiments/exp_002_phase2_boundaries/aggregated_results.json
  - logs/by_prompt/*/config/model/*.yaml (raw samples)
  - Phase 1 hardcoded scores from RESEARCH_STATE.md

Produces (in article/data/):
  - phase1_layer_scores.csv
  - phase2_fine_grained.csv
  - qwen_combos.csv
  - amplification.csv
  - layer_amp_combos.csv
  - prompt_boundaries.csv
  - samples.csv
"""

import csv
import json
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "article" / "data"
AGGREGATED = PROJECT_ROOT / "experiments" / "exp_002_phase2_boundaries" / "aggregated_results.json"
LOGS_DIR = PROJECT_ROOT / "logs" / "by_prompt"

SCORE_COLS = ["sarcasm", "wit", "cynicism", "exaggeration", "meta"]
SCORE_KEYS = [
    "sarcasm_intensity",
    "wit_playfulness",
    "cynicism_negativity",
    "exaggeration_stakes",
    "meta_awareness",
]

MODEL_NAME_MAP = {
    "llama31_8B_Instruct": "llama",
    "llama31_8B_exp2a": "llama",
    "llama31_8B_exp2d": "llama",
    "llama31_8B_exp2d_full": "llama",
    "llama31_8B_exp2e": "llama",
    "llama31_8B_exp2g": "llama",
    "gemma3_4B_it": "gemma",
    "gemma3_4B_exp2b": "gemma",
    "gemma3_4B_exp2d": "gemma",
    "gemma3_4B_exp2d_full": "gemma",
    "gemma3_4B_exp2e": "gemma",
    "gemma3_4B_exp2g": "gemma",
    "qwen25_7B_Instruct": "qwen",
    "qwen25_7B_exp2c": "qwen",
    "qwen25_7B_exp2d": "qwen",
    "qwen25_7B_exp2d_full": "qwen",
    "qwen25_7B_exp2e": "qwen",
}


def load_aggregated():
    """Load the aggregated results JSON."""
    with open(AGGREGATED) as f:
        return json.load(f)


def extract_scores(mean_scores: dict) -> list[float]:
    """Extract score values in canonical order, rounding to 2 decimals."""
    return [round(mean_scores[k], 2) for k in SCORE_KEYS]


def write_csv(path: Path, header: list[str], rows: list[list]):
    """Write rows to CSV with given header."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  Wrote {path.name}: {len(rows)} rows")


# ---------------------------------------------------------------------------
# Phase 1 (hardcoded from RESEARCH_STATE.md)
# ---------------------------------------------------------------------------

def make_phase1():
    """Produce phase1_layer_scores.csv with hardcoded aggregated Phase 1 data."""
    configs = ["base", "full", "0-20", "20-40", "40-60", "60-80", "80-100"]

    llama_scores = [
        (0.1, 0.9, 0.4, 0.2, 1.2),
        (8.9, 8.1, 7.3, 7.9, 5.2),
        (5.4, 5.3, 4.3, 4.3, 2.9),
        (4.3, 4.4, 3.4, 3.4, 3.3),
        (2.1, 2.8, 1.8, 1.7, 2.8),
        (0.7, 1.6, 1.1, 1.0, 1.3),
        (0.3, 1.0, 0.7, 0.4, 1.3),
    ]
    gemma_scores = [
        (0.7, 2.8, 1.2, 1.0, 0.8),
        (9.0, 7.3, 8.0, 7.3, 6.0),
        (1.2, 3.0, 1.5, 1.3, 2.5),
        (2.8, 4.0, 3.0, 2.7, 2.5),
        (5.5, 6.2, 4.0, 5.0, 2.8),
        (2.2, 3.8, 1.8, 1.7, 1.0),
        (2.0, 3.5, 1.5, 1.7, 1.0),
    ]

    n_map = {"base": 9, "full": 9, "0-20": 9, "20-40": 9, "40-60": 9, "60-80": 9, "80-100": 9}

    rows = []
    for cfg, scores in zip(configs, llama_scores):
        rows.append(["llama", cfg, *scores, n_map[cfg]])
    for cfg, scores in zip(configs, gemma_scores):
        rows.append(["gemma", cfg, *scores, n_map[cfg]])

    write_csv(
        DATA_DIR / "phase1_layer_scores.csv",
        ["model", "config"] + SCORE_COLS + ["n"],
        rows,
    )


# ---------------------------------------------------------------------------
# Phase 2 fine-grained layers (10% slices)
# ---------------------------------------------------------------------------

def parse_layer_range(config_name: str) -> tuple[int, int]:
    """Extract (start, end) from config like 'sarcasm_layers_30_40'."""
    m = re.search(r"layers_(\d+)_(\d+)$", config_name)
    assert m, f"Cannot parse layer range from {config_name}"
    return int(m.group(1)), int(m.group(2))


def make_phase2(data: dict):
    """Produce phase2_fine_grained.csv from fine_grained_layers section."""
    rows = []
    for entry in data["fine_grained_layers"].values():
        model = entry["group"]["model"]
        config = entry["group"]["config"]
        start, end = parse_layer_range(config)
        scores = extract_scores(entry["mean_scores"])
        rows.append([model, config, start, end, *scores, entry["n"]])

    rows.sort(key=lambda r: (r[0], r[2]))
    write_csv(
        DATA_DIR / "phase2_fine_grained.csv",
        ["model", "config", "layer_start", "layer_end"] + SCORE_COLS + ["n"],
        rows,
    )


# ---------------------------------------------------------------------------
# Qwen combos
# ---------------------------------------------------------------------------

def make_qwen_combos(data: dict):
    """Produce qwen_combos.csv from qwen_combos section."""
    rows = []
    for entry in data["qwen_combos"].values():
        config = entry["group"]["config"]
        scores = extract_scores(entry["mean_scores"])
        rows.append([config, *scores, entry["n"]])

    write_csv(
        DATA_DIR / "qwen_combos.csv",
        ["config"] + SCORE_COLS + ["n"],
        rows,
    )


# ---------------------------------------------------------------------------
# Amplification
# ---------------------------------------------------------------------------

def parse_multiplier(config_name: str) -> float:
    """Extract multiplier from config like 'sarcasm_full_1_5x' -> 1.5."""
    m = re.search(r"_(\d+)_(\d+)x$", config_name)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"_(\d+)x$", config_name)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse multiplier from {config_name}")


def make_amplification(data: dict):
    """Produce amplification.csv from amplification section plus 1x baselines."""
    rows = []

    # 1x baselines from Phase 1
    rows.append(["llama", 1.0, 8.9, 8.1, 7.3, 7.9, 5.2, 9])
    rows.append(["gemma", 1.0, 9.0, 7.3, 8.0, 7.3, 6.0, 9])
    rows.append(["qwen", 1.0, 7.4, 7.4, 7.4, 7.4, 7.4, 9])  # approximate

    for entry in data["amplification"].values():
        model = entry["group"]["model"]
        config = entry["group"]["config"]
        mult = parse_multiplier(config)
        scores = extract_scores(entry["mean_scores"])
        rows.append([model, mult, *scores, entry["n"]])

    rows.sort(key=lambda r: (r[0], r[1]))
    write_csv(
        DATA_DIR / "amplification.csv",
        ["model", "multiplier"] + SCORE_COLS + ["n"],
        rows,
    )


# ---------------------------------------------------------------------------
# Layer + amplification combos
# ---------------------------------------------------------------------------

def parse_layer_amp(config_name: str) -> tuple[str | None, str, float] | None:
    """Parse layer-amp config strings into (model, layer_range, multiplier).

    Handles two formats:
    - Aggregated (Gemma): 'sarcasm_layers_0_20_3x' -> (None, '0-20', 3.0)
    - Per-sample (Llama): 'exp2g_llama_layeramp_..._sarcasm_layers_40_60_3x.txt' -> ('llama', '40-60', 3.0)
    """
    # Aggregated format (no model name — these are Gemma)
    m = re.match(r"sarcasm_layers_(\d+)_(\d+)_(\d+)x$", config_name)
    if m:
        return None, f"{m.group(1)}-{m.group(2)}", float(m.group(3))
    # Per-sample format with model name
    m = re.match(r"exp2g_(\w+)_layeramp_.*_sarcasm_layers_(\d+)_(\d+)_(\d+)x\.txt$", config_name)
    if m:
        return m.group(1), f"{m.group(2)}-{m.group(3)}", float(m.group(4))
    return None


EXP003_AGGREGATED = PROJECT_ROOT / "experiments" / "exp_003_llama_layeramp_completion" / "aggregated_scores.json"


def make_layer_amp_combos(data: dict):
    """Produce layer_amp_combos.csv with model column, aggregating per-sample entries."""
    from collections import defaultdict
    buckets: dict[tuple[str, str, float], list[list[float]]] = defaultdict(list)

    # Phase 2 exp2g data (Gemma aggregated + Llama per-sample)
    for entry in data["layer_amp_combos"].values():
        config = entry["group"]["config"]
        parsed = parse_layer_amp(config)
        if parsed is None:
            continue
        model, layer_range, mult = parsed
        model = model or "gemma"  # aggregated entries are Gemma
        scores = extract_scores(entry["mean_scores"])
        n = entry["n"]
        for _ in range(n):
            buckets[(model, layer_range, mult)].append(scores)

    # Exp 003: additional Llama layer-amp + full sarcasm amplified data
    if EXP003_AGGREGATED.exists():
        with open(EXP003_AGGREGATED) as f:
            exp003 = json.load(f)
        for config_name, entry in exp003.items():
            parsed = parse_layer_amp(config_name)
            if parsed is None:
                continue
            _, layer_range, mult = parsed
            # All exp003 data is Llama
            for prompt_data in entry.get("per_prompt", {}).values():
                scores = [round(prompt_data[k], 2) for k in SCORE_KEYS]
                buckets[("llama", layer_range, mult)].append(scores)

    rows = []
    for (model, layer_range, mult), score_lists in buckets.items():
        n = len(score_lists)
        mean_scores = [sum(s[i] for s in score_lists) / n for i in range(len(SCORE_COLS))]
        rows.append([model, layer_range, mult, *mean_scores, n])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(
        DATA_DIR / "layer_amp_combos.csv",
        ["model", "layer_range", "multiplier"] + SCORE_COLS + ["n"],
        rows,
    )


# ---------------------------------------------------------------------------
# Prompt boundaries
# ---------------------------------------------------------------------------

def parse_prompt_boundary(prompt_field: str) -> tuple[str, str, str]:
    """Parse prompt field into (model, prompt_type, condition).

    Patterns:
      'gemma_boundary_anti-sarcasm-request_base' -> ('gemma', 'anti-sarcasm-request', 'base')
      'gemma_full_emotional-celebration' -> ('gemma', 'emotional-celebration', 'full')
      'llama_boundary_prefill-' -> ('llama', 'prefill-sarcasm', 'base')  [special case: truncated]
      'llama_full_prefill-' -> ('llama', 'prefill-sarcasm', 'full')
    """
    # boundary entries end with _base
    m = re.match(r"(\w+)_boundary_(.+?)_base$", prompt_field)
    if m:
        model = m.group(1)
        prompt_type = m.group(2)
        if prompt_type == "prefill-":
            prompt_type = "prefill-sarcasm"
        return model, prompt_type, "base"

    # full entries
    m = re.match(r"(\w+)_full_(.+)$", prompt_field)
    if m:
        model = m.group(1)
        prompt_type = m.group(2)
        if prompt_type == "prefill-":
            prompt_type = "prefill-sarcasm"
        return model, prompt_type, "full"

    # edge case: boundary without _base suffix (e.g. 'gemma_boundary_prefill-')
    m = re.match(r"(\w+)_boundary_(.+)$", prompt_field)
    if m:
        model = m.group(1)
        prompt_type = m.group(2)
        if prompt_type == "prefill-":
            prompt_type = "prefill-sarcasm"
        return model, prompt_type, "base"

    raise ValueError(f"Cannot parse prompt boundary: {prompt_field}")


def make_prompt_boundaries(data: dict):
    """Produce prompt_boundaries.csv from prompt_boundaries section."""
    rows = []
    for entry in data["prompt_boundaries"].values():
        prompt_field = entry["group"]["prompt"]
        model_raw = entry["group"]["model"]
        model, prompt_type, condition = parse_prompt_boundary(prompt_field)
        assert model == model_raw, f"Model mismatch: {model} != {model_raw}"
        scores = extract_scores(entry["mean_scores"])
        rows.append([model, prompt_type, condition, *scores])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(
        DATA_DIR / "prompt_boundaries.csv",
        ["model", "prompt_type", "condition"] + SCORE_COLS,
        rows,
    )


# ---------------------------------------------------------------------------
# Samples (raw model outputs)
# ---------------------------------------------------------------------------

def extract_prompt_name(prompt_dir_name: str) -> str:
    """Extract clean prompt name from directory like 'direct-mondays_28673087'."""
    return re.sub(r"_[a-f0-9]+$", "", prompt_dir_name)


def canonical_model(model_dir_or_yaml_field: str) -> str:
    """Map model identifier to canonical short name."""
    return MODEL_NAME_MAP.get(model_dir_or_yaml_field, model_dir_or_yaml_field)


def make_samples():
    """Produce samples.csv by walking logs/by_prompt/*/config/model/*.yaml."""
    rows = []
    yaml_files = sorted(LOGS_DIR.glob("*/*/*/**.yaml"))
    yaml_files = [f for f in yaml_files if not f.name.endswith(".debug.yaml")]

    print(f"  Found {len(yaml_files)} sample YAML files")

    for yf in yaml_files:
        with open(yf) as f:
            doc = yaml.safe_load(f)

        completions = doc.get("completions", [])
        if not completions:
            continue

        text = completions[0]
        config = doc.get("config", "unknown")
        model_raw = doc.get("model", "unknown")
        model = canonical_model(model_raw)

        # prompt_name from directory structure
        # path: logs/by_prompt/<prompt_dir>/<config>/<model_dir>/<file>.yaml
        prompt_dir = yf.parent.parent.parent.name
        prompt_name = extract_prompt_name(prompt_dir)

        rows.append([model, config, prompt_name, text])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(
        DATA_DIR / "samples.csv",
        ["model", "config", "prompt_name", "text"],
        rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = load_aggregated()

    print("Generating CSVs...")
    make_phase1()
    make_phase2(data)
    make_qwen_combos(data)
    make_amplification(data)
    make_layer_amp_combos(data)
    make_prompt_boundaries(data)
    make_samples()
    print("Done.")


if __name__ == "__main__":
    main()
