#!/usr/bin/env python3
"""Prepare v2 data exports from rejudged Phase 1 samples for the Quarto report.

Reads:
  - experiments/exp_001_phase1_replication/rejudge_qwen/judgments/*.json  (new scores)
  - experiments/exp_001_phase1_replication/judging/batch_*/judgments/*.yaml  (old scores)
  - experiments/exp_001_phase1_replication/judging/batch_*/samples/*.txt  (sample text)

Produces (in article/data/):
  - v2_judgments.parquet          -- one row per sample, all scores + text
  - v2_aggregated.parquet         -- aggregated by (model, config)
  - v2_by_prompt.parquet          -- aggregated by (model, config, prompt)
  - v2_dimension_correlations.csv -- pairwise Pearson correlations per model
"""

import json
import re
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "article" / "data"
NEW_JUDGMENTS_DIR = (
    PROJECT_ROOT / "experiments" / "exp_001_phase1_replication" / "rejudge_qwen" / "judgments"
)
OLD_JUDGING_DIR = PROJECT_ROOT / "experiments" / "exp_001_phase1_replication" / "judging"

SCORE_KEYS = [
    "sarcasm_intensity",
    "wit_playfulness",
    "cynicism_negativity",
    "exaggeration_stakes",
    "meta_awareness",
]
SCORE_SHORT = ["sarcasm", "wit", "cynicism", "exaggeration", "meta"]
SCORE_KEY_TO_SHORT = dict(zip(SCORE_KEYS, SCORE_SHORT))

PHASE1_MODEL_PREFIXES = [
    "llama31_8B_Instruct",
    "gemma3_4B_it",
    "qwen25_7B_Instruct",
]
MODEL_NAME_MAP = {
    "llama31_8B_Instruct": "llama",
    "gemma3_4B_it": "gemma",
    "qwen25_7B_Instruct": "qwen",
}

CONFIG_TO_LAYER_RANGE = {
    "base": "base",
    "sarcasm_full": "full",
    "sarcasm_layers_0_20": "0-20",
    "sarcasm_layers_20_40": "20-40",
    "sarcasm_layers_40_60": "40-60",
    "sarcasm_layers_60_80": "60-80",
    "sarcasm_layers_80_100": "80-100",
}


def parse_filename(stem: str) -> tuple[str, str, str]:
    """Parse judgment/sample filename stem into (model, prompt, config).

    Llama:  llama31_8B_Instruct_{prompt}_{config}
    Gemma:  gemma3_4B_it_{prompt}_{config}
    Qwen:   qwen25_7B_Instruct_{prompt}_{hash}_{config}
    """
    for prefix in PHASE1_MODEL_PREFIXES:
        if stem.startswith(prefix + "_"):
            rest = stem[len(prefix) + 1 :]
            model = MODEL_NAME_MAP[prefix]
            break
    else:
        raise ValueError(f"Unknown model prefix in filename: {stem}")

    config_patterns = [
        r"^(.+?)_(sarcasm_layers_\d+_\d+)$",
        r"^(.+?)_(sarcasm_full)$",
        r"^(.+?)_(base)$",
    ]
    for pattern in config_patterns:
        m = re.match(pattern, rest)
        if m:
            prompt_raw = m.group(1)
            config = m.group(2)
            prompt = re.sub(r"_[a-f0-9]{8}$", "", prompt_raw)
            return model, prompt, config

    raise ValueError(f"Cannot parse config from filename rest: {rest} (stem: {stem})")


def load_new_judgment(path: Path) -> dict:
    """Load a new judgment JSON file."""
    with open(path) as f:
        data = json.load(f)
    scores = {SCORE_KEY_TO_SHORT[k]: data["scores"][k] for k in SCORE_KEYS}
    scores["dominant_tone"] = data["qualitative"]["dominant_tone"]
    scores["summary"] = data["qualitative"]["summary"]
    return scores


def load_old_judgment(path: Path) -> dict:
    """Load an old judgment YAML file, returning scores only.

    Some YAMLs have malformed qualitative sections; we parse only the scores
    block when full parsing fails.
    """
    with open(path) as f:
        text = f.read()
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        scores_text = text.split("qualitative:")[0]
        doc = yaml.safe_load(scores_text)

    return {f"old_{SCORE_KEY_TO_SHORT[k]}": doc["scores"][k] for k in SCORE_KEYS}


def build_old_judgment_index() -> dict[str, Path]:
    """Build a map from stem -> path for old judgment YAMLs.

    Later batches override earlier ones (for the one duplicate).
    """
    index = {}
    for batch_dir in sorted(OLD_JUDGING_DIR.glob("batch_*")):
        judgments_dir = batch_dir / "judgments"
        if not judgments_dir.exists():
            continue
        for jf in sorted(judgments_dir.glob("*.yaml")):
            index[jf.stem] = jf
    return index


def build_sample_text_index() -> dict[str, Path]:
    """Build a map from stem -> path for sample text files.

    Later batches override earlier ones.
    """
    index = {}
    for batch_dir in sorted(OLD_JUDGING_DIR.glob("batch_*")):
        samples_dir = batch_dir / "samples"
        if not samples_dir.exists():
            continue
        for sf in sorted(samples_dir.glob("*.txt")):
            index[sf.stem] = sf
    return index


def extract_response_text(path: Path, model: str) -> str:
    """Load sample text, stripping metadata headers for Llama/Gemma.

    Llama/Gemma samples have headers ending with '## Response\\n'.
    Qwen samples are raw text.
    """
    text = path.read_text()
    if model in ("llama", "gemma"):
        marker = "## Response\n"
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker) :].strip()
    return text.strip()


def make_judgments_df() -> pd.DataFrame:
    """Build the main per-sample DataFrame."""
    old_index = build_old_judgment_index()
    text_index = build_sample_text_index()

    rows = []
    for jf in sorted(NEW_JUDGMENTS_DIR.glob("*.json")):
        stem = jf.stem
        model, prompt, config = parse_filename(stem)
        layer_range = CONFIG_TO_LAYER_RANGE[config]

        new_scores = load_new_judgment(jf)

        old_path = old_index.get(stem)
        assert old_path is not None, f"Missing old judgment for {stem}"
        old_scores = load_old_judgment(old_path)

        text_path = text_index.get(stem)
        assert text_path is not None, f"Missing sample text for {stem}"
        text = extract_response_text(text_path, model)

        row = {
            "model": model,
            "prompt": prompt,
            "config": config,
            "layer_range": layer_range,
            **{k: new_scores[k] for k in SCORE_SHORT},
            **old_scores,
            "dominant_tone": new_scores["dominant_tone"],
            "summary": new_scores["summary"],
            "text": text,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "config", "prompt"]).reset_index(drop=True)
    print(f"  v2_judgments: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Configs: {sorted(df['config'].unique())}")
    return df


def make_aggregated(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by (model, config): mean, std, n for each dimension."""
    agg_funcs = {}
    for col in SCORE_SHORT:
        agg_funcs[col] = ["mean", "std"]
    agg_funcs[SCORE_SHORT[0]] = ["mean", "std", "count"]

    grouped = df.groupby(["model", "config"]).agg(agg_funcs)
    grouped.columns = [
        f"{col}_{stat}" if stat != "count" else "n" for col, stat in grouped.columns
    ]
    grouped = grouped.round(2).reset_index()
    grouped = grouped.sort_values(["model", "config"]).reset_index(drop=True)
    print(f"  v2_aggregated: {len(grouped)} rows")
    return grouped


def make_by_prompt(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by (model, config, prompt): mean of each dimension, n."""
    agg_funcs = {col: "mean" for col in SCORE_SHORT}
    agg_funcs[SCORE_SHORT[0]] = ["mean", "count"]

    grouped = df.groupby(["model", "config", "prompt"]).agg(agg_funcs)
    grouped.columns = [
        f"{col}_{stat}" if stat != "count" else "n" for col, stat in grouped.columns
    ]
    # Since each (model, config, prompt) has exactly 1 sample in Phase 1,
    # mean == raw score and n == 1. But keep structure for generality.
    grouped = grouped.round(2).reset_index()
    grouped = grouped.sort_values(["model", "config", "prompt"]).reset_index(drop=True)
    print(f"  v2_by_prompt: {len(grouped)} rows")
    return grouped


def make_dimension_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlations between dimensions, per model."""
    rows = []
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model][SCORE_SHORT]
        corr = model_df.corr(method="pearson")
        for i, d1 in enumerate(SCORE_SHORT):
            for j, d2 in enumerate(SCORE_SHORT):
                if j > i:
                    rows.append({
                        "model": model,
                        "dim1": d1,
                        "dim2": d2,
                        "pearson_r": round(corr.loc[d1, d2], 3),
                    })
    corr_df = pd.DataFrame(rows)
    print(f"  v2_dimension_correlations: {len(corr_df)} rows")
    return corr_df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Building v2 data exports...")
    df = make_judgments_df()

    agg_df = make_aggregated(df)
    by_prompt_df = make_by_prompt(df)
    corr_df = make_dimension_correlations(df)

    df.to_parquet(DATA_DIR / "v2_judgments.parquet", index=False)
    print(f"  Wrote v2_judgments.parquet")

    agg_df.to_parquet(DATA_DIR / "v2_aggregated.parquet", index=False)
    print(f"  Wrote v2_aggregated.parquet")

    by_prompt_df.to_parquet(DATA_DIR / "v2_by_prompt.parquet", index=False)
    print(f"  Wrote v2_by_prompt.parquet")

    corr_df.to_csv(DATA_DIR / "v2_dimension_correlations.csv", index=False)
    print(f"  Wrote v2_dimension_correlations.csv")

    print("Done.")


if __name__ == "__main__":
    main()
