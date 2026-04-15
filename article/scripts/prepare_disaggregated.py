#!/usr/bin/env python3
"""Prepare per-prompt disaggregated data CSVs for the Quarto report.

Reads judgment YAML files from Phase 1 and Phase 2 experiments and produces
disaggregated CSVs (one row per prompt, not aggregated across prompts).

Also produces a curated diverse_samples.csv for inline examples.

Produces (in article/data/):
  - phase1_by_prompt.csv
  - phase2_by_prompt.csv
  - prompt_boundaries_by_model.csv
  - amplification_by_prompt.csv
  - diverse_samples.csv
"""

import csv
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "article" / "data"
PHASE1_JUDGING = PROJECT_ROOT / "experiments" / "exp_001_phase1_replication" / "judging"
PHASE2_JUDGING = PROJECT_ROOT / "experiments" / "exp_002_phase2_boundaries" / "judging"
LOGS_BY_PROMPT = PROJECT_ROOT / "logs" / "by_prompt"
LOGS_BY_REQUEST = PROJECT_ROOT / "logs" / "by_request"

SCORE_KEYS = [
    "sarcasm_intensity",
    "wit_playfulness",
    "cynicism_negativity",
    "exaggeration_stakes",
    "meta_awareness",
]
SCORE_COLS = ["sarcasm", "wit", "cynicism", "exaggeration", "meta"]

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


def write_csv(path: Path, header: list[str], rows: list[list]):
    """Write rows to CSV with given header."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path.name}: {len(rows)} rows")


def load_judgment(path: Path) -> dict:
    """Load a judgment YAML file and return the scores dict.

    Some judgment files have malformed qualitative sections (unescaped quotes).
    Since we only need the scores, we fall back to parsing just the scores block
    (lines before 'qualitative:') when full parsing fails.
    """
    with open(path) as f:
        text = f.read()

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        # Parse only the scores section (everything before 'qualitative:')
        scores_text = text.split("qualitative:")[0]
        doc = yaml.safe_load(scores_text)
        print(f"  WARNING: Partial parse (scores only) for {path.name}")

    return {k: round(float(doc["scores"][k]), 1) for k in SCORE_KEYS}


def extract_prompt_name(dirname: str) -> str:
    """Strip hash suffix from prompt directory name."""
    return re.sub(r"_[a-f0-9]+$", "", dirname)


# ---------------------------------------------------------------------------
# Phase 1: parse filenames like
#   llama31_8B_Instruct_creative-morning-routine_base.yaml
#   qwen25_7B_Instruct_creative-morning-routine_188ac0f8_base.yaml
# ---------------------------------------------------------------------------

# Known model prefixes in Phase 1 judgment filenames
PHASE1_MODEL_PREFIXES = [
    "llama31_8B_Instruct",
    "gemma3_4B_it",
    "qwen25_7B_Instruct",
]


def parse_phase1_filename(stem: str) -> tuple[str, str, str]:
    """Parse Phase 1 judgment filename stem into (model, prompt_name, config).

    Returns canonical model name, clean prompt name, config string.
    """
    for prefix in PHASE1_MODEL_PREFIXES:
        if stem.startswith(prefix + "_"):
            rest = stem[len(prefix) + 1:]
            break
    else:
        raise ValueError(f"Unknown model prefix in Phase 1 filename: {stem}")

    model = MODEL_NAME_MAP[prefix]

    # Qwen files may have a hash suffix in the prompt name: creative-morning-routine_188ac0f8_base
    # We need to find where the config starts. Configs are: base, sarcasm_full, sarcasm_layers_X_Y
    config_patterns = [
        (r"^(.+?)_(sarcasm_layers_\d+_\d+)$", 2),
        (r"^(.+?)_(sarcasm_full)$", 2),
        (r"^(.+?)_(base)$", 2),
    ]
    for pattern, group_idx in config_patterns:
        m = re.match(pattern, rest)
        if m:
            prompt_raw = m.group(1)
            config = m.group(group_idx)
            # Strip hash suffix from prompt name
            prompt_name = re.sub(r"_[a-f0-9]{8}$", "", prompt_raw)
            return model, prompt_name, config

    raise ValueError(f"Cannot parse config from Phase 1 filename rest: {rest}")


def make_phase1_by_prompt():
    """Produce phase1_by_prompt.csv from Phase 1 judgment YAMLs."""
    rows = []
    for batch_dir in sorted(PHASE1_JUDGING.glob("batch_*")):
        judgments_dir = batch_dir / "judgments"
        if not judgments_dir.exists():
            continue
        for jf in sorted(judgments_dir.glob("*.yaml")):
            scores = load_judgment(jf)
            model, prompt_name, config = parse_phase1_filename(jf.stem)
            rows.append([
                model, config, prompt_name,
                *[scores[k] for k in SCORE_KEYS],
            ])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(
        DATA_DIR / "phase1_by_prompt.csv",
        ["model", "config", "prompt_name"] + SCORE_COLS,
        rows,
    )


# ---------------------------------------------------------------------------
# Phase 2: parse filenames like
#   exp2a_llama_fine_creative-morning-routine_sarcasm_layers_0_10.yaml
#   exp2e_gemma_amplify_creative-reddit_sarcasm_full_1_5x.yaml
#   exp2d_llama_boundary_emotional-grief_base.yaml
#   exp2g_llama_layeramp_direct-how-are-you_sarcasm_layers_0_20_2x.txt.yaml
# ---------------------------------------------------------------------------

# Phase 2 filename prefix patterns: exp2X_model_type_
PHASE2_PREFIX_RE = re.compile(
    r"^(exp2[a-z])_(llama|gemma|qwen)_(\w+?)_"
)

# Model short names in phase 2 filenames
PHASE2_MODEL_MAP = {
    "llama": "llama",
    "gemma": "gemma",
    "qwen": "qwen",
}


def parse_phase2_filename(stem: str) -> dict:
    """Parse Phase 2 judgment filename stem into metadata dict.

    Returns dict with keys: exp, model, exp_type, prompt_name, config.
    """
    # Handle .txt suffix artifact
    clean = stem.replace(".txt", "")

    m = PHASE2_PREFIX_RE.match(clean)
    assert m, f"Cannot parse Phase 2 filename: {stem}"

    exp = m.group(1)
    model = PHASE2_MODEL_MAP[m.group(2)]
    exp_type = m.group(3)
    rest = clean[m.end():]

    # rest is like: creative-morning-routine_sarcasm_layers_0_10
    #          or: creative-reddit_sarcasm_full_1_5x
    #          or: emotional-grief_base
    #          or: creative-morning-routine_sarcasm_layers_0_20_2x

    # Try matching configs from most specific to least
    config_patterns = [
        # layer + amplification: sarcasm_layers_X_Y_Nx
        r"^(.+?)_(sarcasm_layers_\d+_\d+_\d+x)$",
        # layer range: sarcasm_layers_X_Y
        r"^(.+?)_(sarcasm_layers_\d+_\d+)$",
        # combo configs: sarcasm_layers_0_50, sarcasm_layers_50_100, sarcasm_layers_bookends, sarcasm_layers_middle
        r"^(.+?)_(sarcasm_layers_\w+)$",
        # amplification with decimal: sarcasm_full_1_5x, sarcasm_full_0_5x
        r"^(.+?)_(sarcasm_full_\d+_\d+x)$",
        # amplification integer: sarcasm_full_2x, sarcasm_full_3x
        r"^(.+?)_(sarcasm_full_\d+x)$",
        # sarcasm_full
        r"^(.+?)_(sarcasm_full)$",
        # base
        r"^(.+?)_(base)$",
    ]
    for pattern in config_patterns:
        pm = re.match(pattern, rest)
        if pm:
            prompt_name = pm.group(1)
            config = pm.group(2)
            return {
                "exp": exp,
                "model": model,
                "exp_type": exp_type,
                "prompt_name": prompt_name,
                "config": config,
            }

    raise ValueError(f"Cannot parse config from Phase 2 rest: {rest} (stem: {stem})")


def make_phase2_by_prompt():
    """Produce phase2_by_prompt.csv from all Phase 2 judgment YAMLs."""
    rows = []
    for batch_dir in sorted(PHASE2_JUDGING.glob("batch_*")):
        judgments_dir = batch_dir / "judgments"
        if not judgments_dir.exists():
            continue
        for jf in sorted(judgments_dir.glob("*.yaml")):
            scores = load_judgment(jf)
            meta = parse_phase2_filename(jf.stem)
            rows.append([
                meta["model"], meta["config"], meta["prompt_name"],
                *[scores[k] for k in SCORE_KEYS],
            ])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(
        DATA_DIR / "phase2_by_prompt.csv",
        ["model", "config", "prompt_name"] + SCORE_COLS,
        rows,
    )


# ---------------------------------------------------------------------------
# Prompt boundaries by model (check if already per-model)
# ---------------------------------------------------------------------------

def make_prompt_boundaries_by_model():
    """Check prompt_boundaries.csv -- it already has per-model rows, so just copy it."""
    src = DATA_DIR / "prompt_boundaries.csv"
    assert src.exists(), f"Missing {src}"

    with open(src) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Verify it has a 'model' column and multiple models
    assert header[0] == "model", f"Expected 'model' column, got: {header[0]}"
    models = {row[0] for row in rows}
    assert len(models) > 1, f"Expected multiple models, got: {models}"

    write_csv(
        DATA_DIR / "prompt_boundaries_by_model.csv",
        header,
        rows,
    )


# ---------------------------------------------------------------------------
# Amplification by prompt
# ---------------------------------------------------------------------------

def parse_amplification_multiplier(config: str) -> float | None:
    """Extract multiplier from config like 'sarcasm_full_1_5x' -> 1.5."""
    m = re.search(r"_(\d+)_(\d+)x$", config)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"_(\d+)x$", config)
    if m:
        return float(m.group(1))
    return None


def make_amplification_by_prompt():
    """Produce amplification_by_prompt.csv from exp2e Phase 2 judgment files."""
    rows = []
    for batch_dir in sorted(PHASE2_JUDGING.glob("batch_*")):
        judgments_dir = batch_dir / "judgments"
        if not judgments_dir.exists():
            continue
        for jf in sorted(judgments_dir.glob("*.yaml")):
            meta = parse_phase2_filename(jf.stem)
            if meta["exp"] != "exp2e":
                continue

            mult = parse_amplification_multiplier(meta["config"])
            assert mult is not None, f"Cannot parse multiplier from {meta['config']}"

            scores = load_judgment(jf)
            rows.append([
                meta["model"], mult, meta["prompt_name"],
                *[scores[k] for k in SCORE_KEYS],
            ])

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(
        DATA_DIR / "amplification_by_prompt.csv",
        ["model", "multiplier", "prompt_name"] + SCORE_COLS,
        rows,
    )


# ---------------------------------------------------------------------------
# Diverse samples
# ---------------------------------------------------------------------------

# Prompt categorization
PROMPT_CATEGORIES = {
    "creative-morning-routine": "creative",
    "creative-pineapple-pizza": "creative",
    "creative-reddit": "creative",
    "direct-first-job-advice": "direct",
    "direct-how-are-you": "direct",
    "direct-mondays": "direct",
    "instruction-exercise-reasons": "instruction",
    "instruction-movie-summary": "instruction",
    "instruction-photosynthesis": "instruction",
    "anti-sarcasm-request": "boundary",
    "complaining-commute": "boundary",
    "emotional-celebration": "boundary",
    "emotional-grief": "boundary",
    "formal-legal-advice": "boundary",
    "formal-medical": "boundary",
    "ironic-rainy-vacation": "boundary",
    "meta-sarcasm-request": "boundary",
    "prefill-sarcasm": "boundary",
    "prefill-sincere": "boundary",
    "raw-completion-mondays": "boundary",
    "raw-completion-work": "boundary",
    "technical-debugging": "boundary",
}


def find_prompt_dir(prompt_name: str) -> Path | None:
    """Find the prompt directory under logs/by_prompt/ matching the given prompt name."""
    for d in LOGS_BY_PROMPT.iterdir():
        if d.is_dir() and extract_prompt_name(d.name) == prompt_name:
            return d
    return None


def load_first_completion(yaml_path: Path) -> str | None:
    """Load the first completion text from a YAML file."""
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)
    completions = doc.get("completions", [])
    if not completions:
        return None
    return completions[0]


def pick_prompts_for_category(category: str, n: int = 3) -> list[str]:
    """Pick n diverse prompts from a category."""
    prompts = [p for p, c in PROMPT_CATEGORIES.items() if c == category]
    return prompts[:n]


# (model_short_name, config, model_dir_pattern, source)
# source: "by_prompt" or "by_request_dir"
DIVERSE_SAMPLE_SPECS = [
    # Llama base and layer configs
    ("llama", "base", "llama31_8B_Instruct", "by_prompt"),
    ("llama", "sarcasm_full", "llama31_8B_Instruct", "by_prompt"),
    ("llama", "sarcasm_layers_0_20", "llama31_8B_Instruct", "by_prompt"),
    ("llama", "sarcasm_layers_40_60", "llama31_8B_Instruct", "by_prompt"),
    ("llama", "sarcasm_layers_80_100", "llama31_8B_Instruct", "by_prompt"),
    ("llama", "sarcasm_full_3x", "llama31_8B_exp2e", "by_prompt"),
    ("llama", "sarcasm_full_0_5x", "llama31_8B_exp2e", "by_prompt"),
    # Gemma base and layer configs
    ("gemma", "base", "gemma3_4B_it", "by_prompt"),
    ("gemma", "sarcasm_full", "gemma3_4B_it", "by_prompt"),
    ("gemma", "sarcasm_layers_0_20", "gemma3_4B_it", "by_prompt"),
    ("gemma", "sarcasm_layers_40_60", "gemma3_4B_it", "by_prompt"),
    ("gemma", "sarcasm_layers_80_100", "gemma3_4B_it", "by_prompt"),
    ("gemma", "sarcasm_full_3x", "gemma3_4B_exp2e", "by_prompt"),
    # Qwen base and full
    ("qwen", "base", "qwen25_7B_Instruct", "by_prompt"),
    ("qwen", "sarcasm_full", "qwen25_7B_Instruct", "by_prompt"),
    ("qwen", "sarcasm_full_3x", "qwen25_7B_exp2e", "by_prompt"),
]

# Boundary samples from by_request dirs
BOUNDARY_SAMPLE_SPECS = [
    # (model, by_request_dir, config_file, config_name)
    ("llama", "exp2d_llama_full", "sarcasm_full.yaml", "sarcasm_full"),
    ("llama", "exp2d_llama_boundary", "base.yaml", "base"),
    ("gemma", "exp2d_gemma_full", "sarcasm_full.yaml", "sarcasm_full"),
    ("gemma", "exp2d_gemma_boundary", "base.yaml", "base"),
    ("qwen", "exp2d_qwen_full", "sarcasm_full.yaml", "sarcasm_full"),
    ("qwen", "exp2d_qwen_boundary", "base.yaml", "base"),
]


def make_diverse_samples():
    """Produce diverse_samples.csv with curated model outputs."""
    rows = []

    # Prompts to sample from for non-boundary configs
    target_prompts = {
        "creative": ["creative-morning-routine", "creative-pineapple-pizza", "creative-reddit"],
        "direct": ["direct-how-are-you", "direct-mondays", "direct-first-job-advice"],
        "instruction": ["instruction-photosynthesis", "instruction-movie-summary"],
    }

    # Non-boundary samples from by_prompt: one from each category for diversity
    for model_short, config, model_dir, source in DIVERSE_SAMPLE_SPECS:
        for cat, prompts in target_prompts.items():
            for prompt_name in prompts:
                prompt_dir = find_prompt_dir(prompt_name)
                if prompt_dir is None:
                    continue
                config_dir = prompt_dir / config / model_dir
                if not config_dir.exists():
                    continue
                yaml_files = sorted(
                    [f for f in config_dir.glob("*.yaml") if not f.name.endswith(".debug.yaml")]
                )
                if not yaml_files:
                    continue
                text = load_first_completion(yaml_files[0])
                if text is None:
                    continue
                rows.append([model_short, config, prompt_name, text, cat])
                break  # one per category

    # Boundary samples from by_request dirs
    boundary_prompts_to_sample = [
        "emotional-grief", "anti-sarcasm-request", "formal-medical",
    ]
    for model_short, req_dir, config_file, config_name in BOUNDARY_SAMPLE_SPECS:
        req_path = LOGS_BY_REQUEST / req_dir
        if not req_path.exists():
            continue
        collected = 0
        for prompt_name in boundary_prompts_to_sample:
            if collected >= 2:
                break
            # Find matching prompt dir (has hash suffix)
            matching = [d for d in req_path.iterdir()
                        if d.is_dir() and extract_prompt_name(d.name) == prompt_name]
            if not matching:
                continue
            yaml_path = matching[0] / config_file
            if not yaml_path.exists():
                continue
            text = load_first_completion(yaml_path)
            if text is None:
                continue
            rows.append([model_short, config_name, prompt_name, text, "boundary"])
            collected += 1

    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    write_csv(
        DATA_DIR / "diverse_samples.csv",
        ["model", "config", "prompt_name", "text", "category"],
        rows,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating disaggregated CSVs...")
    make_phase1_by_prompt()
    make_phase2_by_prompt()
    make_prompt_boundaries_by_model()
    make_amplification_by_prompt()
    make_diverse_samples()
    print("Done.")


if __name__ == "__main__":
    main()
