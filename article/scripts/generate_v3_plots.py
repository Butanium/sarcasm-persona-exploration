#!/usr/bin/env python3
"""Generate all PNG figures for the v3 sarcasm persona article.

Includes: v2 rejudged data analysis, dimension correlations, prompt heatmaps,
and updated versions of all existing plots using the v2 data where available.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

DATA = Path("article/data")
FIGS = Path("article/figures")
FIGS.mkdir(exist_ok=True)

# ── Style constants ──────────────────────────────────────────────────
COLORS = {"llama": "#1565c0", "gemma": "#c62828", "qwen": "#2e7d32"}
MODEL_NAMES = {"llama": "Llama 3.1 8B", "gemma": "Gemma 3 4B", "qwen": "Qwen 2.5 7B"}
MODEL_ORDER = ["llama", "gemma", "qwen"]
DIM_COLORS = {
    "sarcasm": "#e53935", "wit": "#fb8c00", "cynicism": "#7cb342",
    "exaggeration": "#5c6bc0", "meta": "#ab47bc",
}
DIM_LABELS = {
    "sarcasm": "Sarcasm", "wit": "Wit", "cynicism": "Cynicism",
    "exaggeration": "Exaggeration", "meta": "Meta-awareness",
}
DIMS = ["sarcasm", "wit", "cynicism", "exaggeration", "meta"]
LAYER_ORDER = ["base", "0-20", "20-40", "40-60", "60-80", "80-100", "full"]
TEMPLATE = "plotly_white"

WRITE_OPTS = dict(scale=2, engine="kaleido")


def load_all():
    """Load all datasets."""
    return {
        "v2_judgments": pd.read_parquet(DATA / "v2_judgments.parquet"),
        "v2_agg": pd.read_parquet(DATA / "v2_aggregated.parquet"),
        "v2_by_prompt": pd.read_parquet(DATA / "v2_by_prompt.parquet"),
        "v2_corr": pd.read_csv(DATA / "v2_dimension_correlations.csv"),
        "phase1": pd.read_csv(DATA / "phase1_layer_scores.csv"),
        "phase1_bp": pd.read_csv(DATA / "phase1_by_prompt.csv"),
        "phase2_bp": pd.read_csv(DATA / "phase2_by_prompt.csv"),
        "fine": pd.read_csv(DATA / "phase2_fine_grained.csv"),
        "qwen": pd.read_csv(DATA / "qwen_combos.csv"),
        "amp": pd.read_csv(DATA / "amplification.csv"),
        "amp_bp": pd.read_csv(DATA / "amplification_by_prompt.csv"),
        "layer_amp": pd.read_csv(DATA / "layer_amp_combos.csv"),
        "boundaries": pd.read_csv(DATA / "prompt_boundaries.csv"),
    }


# ═══════════════════════════════════════════════════════════════════════
# 1. V2 REJUDGING VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def fig_rejudge_comparison(data):
    """Scatter plots comparing old vs new judging scores for each dimension."""
    df = data["v2_judgments"]

    fig = make_subplots(
        rows=1, cols=5,
        subplot_titles=[DIM_LABELS[d] for d in DIMS],
        horizontal_spacing=0.04,
    )

    for i, dim in enumerate(DIMS, 1):
        old_col = f"old_{dim}"
        new_col = dim
        for model in MODEL_ORDER:
            sub = df[df["model"] == model]
            fig.add_trace(go.Scatter(
                x=sub[old_col], y=sub[new_col],
                mode="markers", name=MODEL_NAMES[model],
                marker=dict(color=COLORS[model], size=5, opacity=0.6),
                legendgroup=model, showlegend=(i == 1),
            ), row=1, col=i)
        # diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 10], y=[0, 10], mode="lines",
            line=dict(color="#999", dash="dash", width=1),
            showlegend=False,
        ), row=1, col=i)
        fig.update_xaxes(range=[-0.5, 10.5], title_text="Old" if i == 3 else "", row=1, col=i)
        fig.update_yaxes(range=[-0.5, 10.5], title_text="New (v2)" if i == 1 else "", row=1, col=i)

    fig.update_layout(
        height=400, width=1400, template=TEMPLATE,
        title="Rejudging Validation: Old vs New Scores",
        margin=dict(t=60, b=50),
    )
    fig.write_image(FIGS / "fig_rejudge_comparison.png", **WRITE_OPTS)
    print("  wrote fig_rejudge_comparison.png")


def fig_rejudge_shift(data):
    """Bar chart showing mean shift (new - old) per dimension per model."""
    df = data["v2_judgments"]

    shifts = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        for dim in DIMS:
            shift = (sub[dim] - sub[f"old_{dim}"]).mean()
            shifts.append({"model": model, "dim": dim, "shift": shift})
    shifts_df = pd.DataFrame(shifts)

    fig = go.Figure()
    for dim in DIMS:
        sub = shifts_df[shifts_df["dim"] == dim]
        fig.add_trace(go.Bar(
            x=[MODEL_NAMES[m] for m in sub["model"]],
            y=sub["shift"],
            name=DIM_LABELS[dim],
            marker_color=DIM_COLORS[dim],
        ))

    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.update_layout(
        height=400, width=800, template=TEMPLATE, barmode="group",
        title="Rejudging Shift: New − Old Score (per dimension, per model)",
        yaxis_title="Mean score shift",
        margin=dict(t=60, b=40),
    )
    fig.write_image(FIGS / "fig_rejudge_shift.png", **WRITE_OPTS)
    print("  wrote fig_rejudge_shift.png")


# ═══════════════════════════════════════════════════════════════════════
# 2. DIMENSION CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════

def fig_dimension_correlations(data):
    """Correlation heatmaps for each model."""
    corr_df = data["v2_corr"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_NAMES[m] for m in MODEL_ORDER],
        horizontal_spacing=0.08,
    )

    for col_idx, model in enumerate(MODEL_ORDER, 1):
        sub = corr_df[corr_df["model"] == model]
        # Build symmetric matrix
        mat = pd.DataFrame(np.ones((5, 5)), index=DIMS, columns=DIMS)
        for _, row in sub.iterrows():
            mat.loc[row["dim1"], row["dim2"]] = row["pearson_r"]
            mat.loc[row["dim2"], row["dim1"]] = row["pearson_r"]

        labels = [DIM_LABELS[d] for d in DIMS]
        text = [[f"{mat.iloc[i, j]:.2f}" for j in range(5)] for i in range(5)]

        fig.add_trace(go.Heatmap(
            z=mat.values, x=labels, y=labels,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=text, texttemplate="%{text}",
            showscale=(col_idx == 3),
            colorbar=dict(title="r", len=0.8) if col_idx == 3 else None,
        ), row=1, col=col_idx)

    fig.update_layout(
        height=450, width=1400, template=TEMPLATE,
        title="Dimension Correlations (Pearson r, v2 judging)",
        margin=dict(t=60, b=20),
    )
    fig.write_image(FIGS / "fig_dimension_correlations.png", **WRITE_OPTS)
    print("  wrote fig_dimension_correlations.png")


# ═══════════════════════════════════════════════════════════════════════
# 3. V2 PHASE 1: SARCASM BY LAYER (3-MODEL VIEW)
# ═══════════════════════════════════════════════════════════════════════

def fig_v2_phase1_sarcasm(data):
    """Phase 1 sarcasm by layer range using v2 rejudged data, all 3 models."""
    df = data["v2_agg"].copy()
    # Map config to layer_range label
    config_to_layer = {
        "base": "base", "sarcasm_full": "full",
        "sarcasm_layers_0_20": "0-20", "sarcasm_layers_20_40": "20-40",
        "sarcasm_layers_40_60": "40-60", "sarcasm_layers_60_80": "60-80",
        "sarcasm_layers_80_100": "80-100",
    }
    df["layer"] = df["config"].map(config_to_layer)
    df["layer"] = pd.Categorical(df["layer"], categories=LAYER_ORDER, ordered=True)
    df = df.sort_values("layer")

    fig = go.Figure()
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        fig.add_trace(go.Bar(
            x=sub["layer"], y=sub["sarcasm_mean"],
            error_y=dict(type="data", array=sub["sarcasm_std"], visible=True),
            name=MODEL_NAMES[model], marker_color=COLORS[model],
        ))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title="Phase 1: Sarcasm by Layer Range (v2 judging, mean ± std)",
        xaxis_title="Layer Range", yaxis_title="Sarcasm (0-10)",
        yaxis_range=[0, 10.5],
        height=450, width=1000,
    )
    fig.write_image(FIGS / "fig_v2_phase1_sarcasm.png", **WRITE_OPTS)
    print("  wrote fig_v2_phase1_sarcasm.png")


# ═══════════════════════════════════════════════════════════════════════
# 4. V2 ALL DIMENSIONS BY LAYER (HEATMAPS)
# ═══════════════════════════════════════════════════════════════════════

def fig_v2_heatmaps(data):
    """Heatmap of all dimensions × layer ranges for each model (v2 data)."""
    df = data["v2_agg"].copy()
    config_to_layer = {
        "base": "base", "sarcasm_full": "full",
        "sarcasm_layers_0_20": "0-20", "sarcasm_layers_20_40": "20-40",
        "sarcasm_layers_40_60": "40-60", "sarcasm_layers_60_80": "60-80",
        "sarcasm_layers_80_100": "80-100",
    }
    df["layer"] = df["config"].map(config_to_layer)
    df["layer"] = pd.Categorical(df["layer"], categories=LAYER_ORDER, ordered=True)
    df = df.sort_values("layer")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[MODEL_NAMES[m] for m in MODEL_ORDER],
        vertical_spacing=0.1,
    )

    for row_idx, model in enumerate(MODEL_ORDER, 1):
        sub = df[df["model"] == model]
        mean_cols = [f"{d}_mean" for d in DIMS]
        z = sub[mean_cols].values.T
        labels_y = [DIM_LABELS[d] for d in DIMS]
        labels_x = sub["layer"].tolist()
        text = [[f"{v:.1f}" for v in row] for row in z]

        fig.add_trace(go.Heatmap(
            z=z, x=labels_x, y=labels_y,
            colorscale="RdYlBu_r", zmin=0, zmax=10,
            text=text, texttemplate="%{text}",
            showscale=(row_idx == 1),
            colorbar=dict(title="Score", len=0.25, y=0.85) if row_idx == 1 else None,
        ), row=row_idx, col=1)

    fig.update_layout(
        height=700, width=900, template=TEMPLATE,
        title="All Dimensions × Layer Ranges (v2 judging)",
        margin=dict(t=60, b=20),
    )
    fig.write_image(FIGS / "fig_v2_heatmaps.png", **WRITE_OPTS)
    print("  wrote fig_v2_heatmaps.png")


# ═══════════════════════════════════════════════════════════════════════
# 5. V2 ALL DIMENSIONS LINE PLOTS
# ═══════════════════════════════════════════════════════════════════════

def fig_v2_all_dims_lines(data):
    """Line plots of all 5 dimensions by layer range, per model (v2 data)."""
    df = data["v2_agg"].copy()
    config_to_layer = {
        "base": "base", "sarcasm_full": "full",
        "sarcasm_layers_0_20": "0-20", "sarcasm_layers_20_40": "20-40",
        "sarcasm_layers_40_60": "40-60", "sarcasm_layers_60_80": "60-80",
        "sarcasm_layers_80_100": "80-100",
    }
    df["layer"] = df["config"].map(config_to_layer)
    df["layer"] = pd.Categorical(df["layer"], categories=LAYER_ORDER, ordered=True)
    df = df.sort_values("layer")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_NAMES[m] for m in MODEL_ORDER],
        shared_yaxes=True, horizontal_spacing=0.06,
    )

    for col_idx, model in enumerate(MODEL_ORDER, 1):
        sub = df[df["model"] == model]
        for dim in DIMS:
            fig.add_trace(go.Scatter(
                x=sub["layer"], y=sub[f"{dim}_mean"],
                mode="lines+markers",
                name=DIM_LABELS[dim], line=dict(color=DIM_COLORS[dim], width=2),
                marker=dict(size=6),
                legendgroup=dim, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_yaxes(range=[0, 10], title_text="Score (0-10)", row=1, col=1)
    fig.update_layout(
        height=420, width=1400, template=TEMPLATE,
        title="All Dimensions by Layer Range (v2 judging)",
        margin=dict(t=60, b=40),
    )
    fig.write_image(FIGS / "fig_v2_all_dims_lines.png", **WRITE_OPTS)
    print("  wrote fig_v2_all_dims_lines.png")


# ═══════════════════════════════════════════════════════════════════════
# 6. PROMPT × LAYER HEATMAPS (V2 DATA)
# ═══════════════════════════════════════════════════════════════════════

def fig_prompt_layer_heatmaps(data):
    """Heatmap of sarcasm scores: prompt × layer_range, one per model."""
    df = data["v2_judgments"].copy()

    # Map config to short layer label
    config_to_layer = {
        "base": "base", "sarcasm_full": "full",
        "sarcasm_layers_0_20": "0-20", "sarcasm_layers_20_40": "20-40",
        "sarcasm_layers_40_60": "40-60", "sarcasm_layers_60_80": "60-80",
        "sarcasm_layers_80_100": "80-100",
    }
    df["layer"] = df["config"].map(config_to_layer)
    df["layer"] = pd.Categorical(df["layer"], categories=LAYER_ORDER, ordered=True)

    # Shorter prompt labels
    prompt_short = {
        "creative-morning-routine": "morning routine",
        "creative-pineapple-pizza": "pineapple pizza",
        "creative-reddit": "reddit post",
        "direct-first-job-advice": "first job",
        "direct-how-are-you": "how are you",
        "direct-mondays": "mondays",
        "instruction-exercise-reasons": "exercise",
        "instruction-movie-summary": "movie summary",
        "instruction-photosynthesis": "photosynthesis",
    }
    df["prompt_short"] = df["prompt"].map(prompt_short)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_NAMES[m] for m in MODEL_ORDER],
        horizontal_spacing=0.08,
    )

    for col_idx, model in enumerate(MODEL_ORDER, 1):
        sub = df[df["model"] == model]
        pivot = sub.pivot_table(index="prompt_short", columns="layer", values="sarcasm", aggfunc="mean")
        pivot = pivot.reindex(columns=LAYER_ORDER)
        # Sort prompts by mean sarcasm across configs
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]

        text = [[f"{v:.0f}" if not np.isnan(v) else "" for v in row] for row in pivot.values]

        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale="RdYlBu_r", zmin=0, zmax=10,
            text=text, texttemplate="%{text}",
            showscale=(col_idx == 3),
            colorbar=dict(title="Sarcasm", len=0.8) if col_idx == 3 else None,
        ), row=1, col=col_idx)

    fig.update_layout(
        height=450, width=1500, template=TEMPLATE,
        title="Sarcasm by Prompt × Layer Range (v2 judging)",
        margin=dict(t=60, b=20, l=120),
    )
    fig.write_image(FIGS / "fig_prompt_layer_heatmaps.png", **WRITE_OPTS)
    print("  wrote fig_prompt_layer_heatmaps.png")


# ═══════════════════════════════════════════════════════════════════════
# 7. AMPLIFICATION – ALL 5 DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════

def fig_amplification_all_dims(data):
    """All 5 dimensions under amplification, one subplot per model."""
    amp = data["amp"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[MODEL_NAMES[m] for m in MODEL_ORDER],
        shared_yaxes=True, horizontal_spacing=0.06,
    )

    for col_idx, model in enumerate(MODEL_ORDER, 1):
        sub = amp[amp["model"] == model].sort_values("multiplier")
        for dim in DIMS:
            fig.add_trace(go.Scatter(
                x=sub["multiplier"], y=sub[dim],
                mode="lines+markers",
                name=DIM_LABELS[dim], line=dict(color=DIM_COLORS[dim], width=2),
                marker=dict(size=6),
                legendgroup=dim, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_xaxes(title_text="Multiplier", row=1, col=2)
    fig.update_yaxes(range=[0, 10], title_text="Score (0-10)", row=1, col=1)
    fig.update_layout(
        height=420, width=1400, template=TEMPLATE,
        title="All Dimensions Under Amplification",
        margin=dict(t=60, b=50),
    )
    fig.write_image(FIGS / "fig_amplification_all_dims.png", **WRITE_OPTS)
    print("  wrote fig_amplification_all_dims.png")


# ═══════════════════════════════════════════════════════════════════════
# 8. AMPLIFICATION – SARCASM VS WIT (UPDATED WITH BETTER STYLING)
# ═══════════════════════════════════════════════════════════════════════

def fig_amplification_sarcasm_wit(data):
    """Sarcasm and wit by amplification, with shaded sweet-spot region."""
    amp = data["amp"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sarcasm Intensity", "Wit / Playfulness"),
        shared_yaxes=True, horizontal_spacing=0.06,
    )

    for col_idx, dim in enumerate(["sarcasm", "wit"], 1):
        # Sweet spot shading
        fig.add_vrect(x0=1.25, x1=2.25, fillcolor="#e8f5e9", opacity=0.5,
                      line_width=0, row=1, col=col_idx)
        for model in MODEL_ORDER:
            sub = amp[amp["model"] == model].sort_values("multiplier")
            fig.add_trace(go.Scatter(
                x=sub["multiplier"], y=sub[dim],
                mode="lines+markers",
                name=MODEL_NAMES[model],
                line=dict(color=COLORS[model], width=2.5),
                marker=dict(size=8),
                legendgroup=model, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_xaxes(title_text="Amplification multiplier")
    fig.update_yaxes(range=[0, 10], title_text="Score (0-10)", row=1, col=1)
    fig.update_layout(
        height=400, width=1000, template=TEMPLATE,
        title="Amplification: Sarcasm vs Wit (green = sweet spot 1.5-2x)",
        margin=dict(t=60, b=50),
    )
    fig.write_image(FIGS / "fig_v2_amplification.png", **WRITE_OPTS)
    print("  wrote fig_v2_amplification.png")


# ═══════════════════════════════════════════════════════════════════════
# 9. FINE-GRAINED 10% SLICES (UPDATED)
# ═══════════════════════════════════════════════════════════════════════

def fig_fine_grained(data):
    """10% layer slices for Llama and Gemma with all dimensions."""
    fine = data["fine"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Llama 3.1 8B (0-40% region)", "Gemma 3 4B (30-70% region)"),
        shared_yaxes=True, horizontal_spacing=0.08,
    )

    for col_idx, model in enumerate(["llama", "gemma"], 1):
        sub = fine[fine["model"] == model].sort_values("layer_start")
        labels = [f"{int(r.layer_start)}-{int(r.layer_end)}%" for _, r in sub.iterrows()]
        for dim in ["sarcasm", "wit", "cynicism"]:
            fig.add_trace(go.Scatter(
                x=labels, y=sub[dim],
                mode="lines+markers",
                name=DIM_LABELS[dim],
                line=dict(color=DIM_COLORS[dim], width=2.5 if dim == "sarcasm" else 1.5),
                marker=dict(size=8 if dim == "sarcasm" else 5),
                legendgroup=dim, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_yaxes(range=[0, 5.5], title_text="Score (zoomed 0-5)", row=1, col=1)
    fig.update_layout(
        height=400, width=1000, template=TEMPLATE,
        title="Fine-Grained 10% Layer Slices: Sarcasm, Wit, Cynicism",
        margin=dict(t=60, b=40),
    )
    fig.write_image(FIGS / "fig_v2_fine_grained.png", **WRITE_OPTS)
    print("  wrote fig_v2_fine_grained.png")


# ═══════════════════════════════════════════════════════════════════════
# 10. LAYER × AMPLIFICATION INTERACTION (UPDATED)
# ═══════════════════════════════════════════════════════════════════════

def fig_layer_amp_interaction(data):
    """Layer selection vs amplification strength — right layers vs wrong layers."""
    layer_amp = data["layer_amp"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sarcasm", "Wit"),
        shared_yaxes=True, horizontal_spacing=0.08,
    )

    line_styles = {"0-20": dict(color="#90a4ae", dash="dash"), "40-60": dict(color="#e53935")}
    for col_idx, dim in enumerate(["sarcasm", "wit"], 1):
        for lr in ["0-20", "40-60"]:
            sub = layer_amp[layer_amp["layer_range"] == lr].sort_values("multiplier")
            label = f"Layers {lr}% ({'wrong' if lr == '0-20' else 'right'})"
            fig.add_trace(go.Scatter(
                x=sub["multiplier"], y=sub[dim],
                mode="lines+markers", name=label,
                line=dict(**line_styles[lr], width=2.5),
                marker=dict(size=8),
                legendgroup=lr, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    # Full LoRA reference line
    fig.add_hline(y=9.0, line_dash="dot", line_color="#999",
                  annotation_text="Full LoRA @ 1x", annotation_position="top right",
                  row=1, col=1)

    fig.update_xaxes(title_text="Amplification multiplier")
    fig.update_yaxes(range=[0, 10.5], title_text="Score (0-10)", row=1, col=1)
    fig.update_layout(
        height=400, width=1000, template=TEMPLATE,
        title="Gemma: Wrong Layers Can't Be Compensated by Amplification",
        margin=dict(t=60, b=50),
    )
    fig.write_image(FIGS / "fig_v2_layer_amp.png", **WRITE_OPTS)
    print("  wrote fig_v2_layer_amp.png")


# ═══════════════════════════════════════════════════════════════════════
# 11. PROMPT ROBUSTNESS (UPDATED)
# ═══════════════════════════════════════════════════════════════════════

def fig_prompt_robustness(data):
    """Paired dot plot: base vs full sarcasm per prompt type, aggregated."""
    boundaries = data["boundaries"]
    agg = boundaries.groupby(["prompt_type", "condition"])["sarcasm"].mean().reset_index()

    base = agg[agg["condition"] == "base"].set_index("prompt_type")
    full = agg[agg["condition"] == "full"].set_index("prompt_type")
    merged = base[["sarcasm"]].rename(columns={"sarcasm": "base"}).join(
        full[["sarcasm"]].rename(columns={"sarcasm": "full"})
    ).dropna()
    merged["delta"] = merged["full"] - merged["base"]
    merged = merged.sort_values("full", ascending=True)

    fig = go.Figure()

    # Connecting lines
    for prompt in merged.index:
        fig.add_trace(go.Scatter(
            x=[merged.loc[prompt, "base"], merged.loc[prompt, "full"]],
            y=[prompt, prompt],
            mode="lines", line=dict(color="#ddd", width=1.5),
            showlegend=False,
        ))

    # Base dots
    fig.add_trace(go.Scatter(
        x=merged["base"], y=merged.index,
        mode="markers", name="Base (no LoRA)",
        marker=dict(color="#90a4ae", size=10, symbol="circle"),
    ))

    # Full dots
    fig.add_trace(go.Scatter(
        x=merged["full"], y=merged.index,
        mode="markers", name="Full sarcasm LoRA",
        marker=dict(color="#e53935", size=10, symbol="diamond"),
    ))

    fig.update_layout(
        height=550, width=900, template=TEMPLATE,
        title="Persona Robustness: Base vs Full LoRA Across Prompt Types",
        xaxis=dict(range=[-0.5, 10.5], title="Sarcasm Intensity (0-10)"),
        margin=dict(l=200, t=60, b=40),
        legend=dict(x=0.65, y=0.1),
    )
    fig.write_image(FIGS / "fig_v2_prompt_robustness.png", **WRITE_OPTS)
    print("  wrote fig_v2_prompt_robustness.png")


# ═══════════════════════════════════════════════════════════════════════
# 12. LOCALIZATION STRENGTH SUMMARY
# ═══════════════════════════════════════════════════════════════════════

def fig_localization_summary(data):
    """Summary figure: layer profiles normalized to show localization strength."""
    df = data["v2_agg"].copy()
    config_to_layer = {
        "sarcasm_layers_0_20": "0-20", "sarcasm_layers_20_40": "20-40",
        "sarcasm_layers_40_60": "40-60", "sarcasm_layers_60_80": "60-80",
        "sarcasm_layers_80_100": "80-100",
    }
    # Only layer slices (not base or full)
    df = df[df["config"].isin(config_to_layer)].copy()
    df["layer"] = df["config"].map(config_to_layer)

    layer_slice_order = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    df["layer"] = pd.Categorical(df["layer"], categories=layer_slice_order, ordered=True)
    df = df.sort_values("layer")

    fig = go.Figure()
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        # Normalize: fraction of total sarcasm across slices
        total = sub["sarcasm_mean"].sum()
        normalized = sub["sarcasm_mean"] / total * 100 if total > 0 else sub["sarcasm_mean"]
        fig.add_trace(go.Scatter(
            x=sub["layer"], y=normalized,
            mode="lines+markers+text",
            name=MODEL_NAMES[model],
            line=dict(color=COLORS[model], width=3),
            marker=dict(size=10),
            text=[f"{v:.0f}%" for v in normalized],
            textposition="top center",
        ))

    fig.add_hline(y=20, line_dash="dash", line_color="#ccc",
                  annotation_text="Uniform (20%)", annotation_position="bottom right")

    fig.update_layout(
        height=420, width=900, template=TEMPLATE,
        title="Localization Profiles: % of Total Sarcasm by Layer Slice",
        xaxis_title="Layer Range",
        yaxis=dict(title="% of total sarcasm (across 5 slices)", range=[0, 55]),
        margin=dict(t=60, b=50),
    )
    fig.write_image(FIGS / "fig_localization_summary.png", **WRITE_OPTS)
    print("  wrote fig_localization_summary.png")


# ═══════════════════════════════════════════════════════════════════════
# 13. QWEN COMBOS (UPDATED)
# ═══════════════════════════════════════════════════════════════════════

def fig_qwen_combos(data):
    """Qwen layer combination strategies."""
    qwen = data["qwen"]
    config_labels = {
        "sarcasm_layers_0_50": "First half\n(0-50%)",
        "sarcasm_layers_50_100": "Second half\n(50-100%)",
        "sarcasm_layers_bookends": "Bookends\n(0-25% + 75-100%)",
        "sarcasm_layers_middle": "Middle\n(25-75%)",
    }
    qwen_plot = qwen.copy()
    qwen_plot["label"] = qwen_plot["config"].map(config_labels)

    fig = go.Figure()
    for dim in DIMS:
        fig.add_trace(go.Bar(
            x=qwen_plot["label"], y=qwen_plot[dim],
            name=DIM_LABELS[dim], marker_color=DIM_COLORS[dim],
        ))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title="Qwen 2.5 7B: Layer Combination Strategies",
        xaxis_title="Configuration", yaxis_title="Score (0-10)",
        yaxis_range=[0, 9],
        height=420, width=900,
    )
    fig.write_image(FIGS / "fig_v2_qwen_combos.png", **WRITE_OPTS)
    print("  wrote fig_v2_qwen_combos.png")


# ═══════════════════════════════════════════════════════════════════════
# 14. DOMINANT TONE STACKED BAR
# ═══════════════════════════════════════════════════════════════════════

def fig_dominant_tone_stacked(data):
    """Stacked bar of dominant_tone distribution by config, aggregated across models."""
    df = data["v2_judgments"].copy()

    config_to_layer = {
        "base": "base", "sarcasm_full": "full",
        "sarcasm_layers_0_20": "0-20%", "sarcasm_layers_20_40": "20-40%",
        "sarcasm_layers_40_60": "40-60%", "sarcasm_layers_60_80": "60-80%",
        "sarcasm_layers_80_100": "80-100%",
    }
    df["layer"] = df["config"].map(config_to_layer)
    layer_order = ["base", "0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "full"]
    df["layer"] = pd.Categorical(df["layer"], categories=layer_order, ordered=True)

    counts = df.groupby(["layer", "dominant_tone"], observed=False).size().unstack(fill_value=0)
    # Convert to percentages
    pcts = counts.div(counts.sum(axis=1), axis=0) * 100

    tone_colors = {
        "sincere": "#90a4ae", "sarcastic": "#c62828",
        "playful": "#7b1fa2", "neutral": "#bdbdbd", "cynical": "#455a64",
    }
    tone_order = ["sincere", "playful", "neutral", "cynical", "sarcastic"]

    fig = go.Figure()
    for tone in tone_order:
        if tone in pcts.columns:
            fig.add_trace(go.Bar(
                x=pcts.index.tolist(), y=pcts[tone],
                name=tone.capitalize(), marker_color=tone_colors[tone],
                text=[f"{v:.0f}%" if v >= 5 else "" for v in pcts[tone]],
                textposition="inside", textfont=dict(color="white", size=11),
            ))

    fig.update_layout(
        barmode="stack",
        height=450, width=900, template=TEMPLATE,
        title="Dominant Tone by Layer Range (aggregated across 3 models)",
        xaxis_title="Layer Range",
        yaxis=dict(title="Percentage of samples", range=[0, 105]),
        legend=dict(orientation="h", y=-0.15, x=0.3),
        margin=dict(t=60, b=80),
    )
    fig.write_image(FIGS / "fig_dominant_tone_stacked.png", **WRITE_OPTS)
    print("  wrote fig_dominant_tone_stacked.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    data = load_all()

    print("\nGenerating v3 figures...")

    # Rejudging validation
    fig_rejudge_comparison(data)
    fig_rejudge_shift(data)

    # Dimension correlations
    fig_dimension_correlations(data)

    # V2 Phase 1
    fig_v2_phase1_sarcasm(data)
    fig_v2_heatmaps(data)
    fig_v2_all_dims_lines(data)

    # Prompt × layer
    fig_prompt_layer_heatmaps(data)

    # Amplification
    fig_amplification_all_dims(data)
    fig_amplification_sarcasm_wit(data)

    # Fine-grained
    fig_fine_grained(data)

    # Layer × amplification
    fig_layer_amp_interaction(data)

    # Prompt robustness
    fig_prompt_robustness(data)

    # Summary
    fig_localization_summary(data)

    # Qwen
    fig_qwen_combos(data)

    # Dominant tone
    fig_dominant_tone_stacked(data)

    print(f"\nDone! All figures in {FIGS}/")
    print(f"Total: {len(list(FIGS.glob('fig_v2_*.png')) + list(FIGS.glob('fig_rejudge_*.png')) + list(FIGS.glob('fig_dimension_*.png')) + list(FIGS.glob('fig_prompt_layer_*.png')) + list(FIGS.glob('fig_localization_*.png')) + list(FIGS.glob('fig_amplification_all_*.png')))} new figures")


if __name__ == "__main__":
    main()
