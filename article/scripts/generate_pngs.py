#!/usr/bin/env python3
"""Generate all PNG figures for the sarcasm persona article."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA = "article/data"
FIGS = "article/figures"

COLORS = {"llama": "#1565c0", "gemma": "#c62828", "qwen": "#2e7d32"}
MODEL_NAMES = {"llama": "Llama 3.1 8B", "gemma": "Gemma 3 4B", "qwen": "Qwen 2.5 7B"}
TEMPLATE = "plotly_white"

LAYER_ORDER = ["base", "0-20", "20-40", "40-60", "60-80", "80-100", "full"]


def fig_phase1_sarcasm():
    """Phase 1 sarcasm by layer range, all 3 models."""
    df = pd.read_csv(f"{DATA}/phase1_layer_scores.csv")
    df["config"] = pd.Categorical(df["config"], categories=LAYER_ORDER, ordered=True)
    df = df.sort_values("config")

    fig = go.Figure()
    for model in ["llama", "gemma", "qwen"]:
        sub = df[df["model"] == model]
        fig.add_trace(go.Bar(
            x=sub["config"], y=sub["sarcasm"],
            name=MODEL_NAMES[model], marker_color=COLORS[model],
        ))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title="Phase 1: Sarcasm Score by Layer Range",
        xaxis_title="Layer Range", yaxis_title="Sarcasm (0-10)",
        yaxis_range=[0, 10],
    )
    fig.write_image(f"{FIGS}/fig_phase1_sarcasm.png", width=1200, height=500)
    print("  wrote fig_phase1_sarcasm.png")


def fig_phase1_all_dims():
    """All 5 dimensions by layer range, all 3 models side-by-side."""
    df = pd.read_csv(f"{DATA}/phase1_layer_scores.csv")
    df["config"] = pd.Categorical(df["config"], categories=LAYER_ORDER, ordered=True)
    df = df.sort_values("config")
    dims = ["sarcasm", "wit", "cynicism", "exaggeration", "meta"]
    dim_colors = ["#1565c0", "#7b1fa2", "#c62828", "#e65100", "#2e7d32"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=["Llama 3.1 8B", "Gemma 3 4B", "Qwen 2.5 7B"])
    for col_idx, model in enumerate(["llama", "gemma", "qwen"], 1):
        sub = df[df["model"] == model]
        for dim, color in zip(dims, dim_colors):
            fig.add_trace(go.Scatter(
                x=sub["config"], y=sub[dim], mode="lines+markers",
                name=dim.capitalize(), marker_color=color,
                legendgroup=dim, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(template=TEMPLATE, title="Phase 1: All Dimensions by Layer Range")
    fig.update_yaxes(range=[0, 10], title_text="Score (0-10)", row=1, col=1)
    fig.update_yaxes(range=[0, 10], row=1, col=2)
    fig.update_yaxes(range=[0, 10], row=1, col=3)
    fig.write_image(f"{FIGS}/fig_phase1_all_dims.png", width=1400, height=500)
    print("  wrote fig_phase1_all_dims.png")


def fig_phase2_fine():
    """Phase 2 fine-grained 10% slices, sarcasm + wit, Llama and Gemma side-by-side."""
    df = pd.read_csv(f"{DATA}/phase2_fine_grained.csv")
    df["label"] = df["layer_start"].astype(str) + "-" + df["layer_end"].astype(str)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Llama 3.1 8B", "Gemma 3 4B"])
    for col_idx, model in enumerate(["llama", "gemma"], 1):
        sub = df[df["model"] == model].sort_values("layer_start")
        fig.add_trace(go.Bar(
            x=sub["label"], y=sub["sarcasm"], name="Sarcasm",
            marker_color="#1565c0", legendgroup="sarcasm", showlegend=(col_idx == 1),
        ), row=1, col=col_idx)
        fig.add_trace(go.Bar(
            x=sub["label"], y=sub["wit"], name="Wit",
            marker_color="#7b1fa2", legendgroup="wit", showlegend=(col_idx == 1),
        ), row=1, col=col_idx)

    fig.update_layout(template=TEMPLATE, barmode="group",
                      title="Phase 2: Fine-Grained 10% Layer Slices")
    fig.update_yaxes(range=[0, 10], title_text="Score (0-10)", row=1, col=1)
    fig.update_yaxes(range=[0, 10], row=1, col=2)
    fig.write_image(f"{FIGS}/fig_phase2_fine.png", width=1200, height=500)
    print("  wrote fig_phase2_fine.png")


def fig_qwen_combos():
    """Qwen layer combinations bar chart."""
    df = pd.read_csv(f"{DATA}/qwen_combos.csv")
    dims = ["sarcasm", "wit", "cynicism", "exaggeration", "meta"]
    dim_colors = ["#1565c0", "#7b1fa2", "#c62828", "#e65100", "#2e7d32"]

    fig = go.Figure()
    for dim, color in zip(dims, dim_colors):
        fig.add_trace(go.Bar(
            x=df["config"], y=df[dim], name=dim.capitalize(), marker_color=color,
        ))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title="Qwen 2.5 7B: Layer Combination Strategies",
        xaxis_title="Configuration", yaxis_title="Score (0-10)",
        yaxis_range=[0, 10],
    )
    fig.write_image(f"{FIGS}/fig_qwen_combos.png", width=800, height=400)
    print("  wrote fig_qwen_combos.png")


def fig_amplification():
    """Sarcasm and wit by amplification multiplier, all 3 models."""
    df = pd.read_csv(f"{DATA}/amplification.csv")

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Sarcasm", "Wit"])
    for model in ["llama", "gemma", "qwen"]:
        sub = df[df["model"] == model].sort_values("multiplier")
        for col_idx, dim in enumerate(["sarcasm", "wit"], 1):
            fig.add_trace(go.Scatter(
                x=sub["multiplier"], y=sub[dim], mode="lines+markers",
                name=MODEL_NAMES[model], marker_color=COLORS[model],
                legendgroup=model, showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(template=TEMPLATE, title="Amplification: Sarcasm and Wit by Multiplier")
    fig.update_xaxes(title_text="Multiplier", row=1, col=1)
    fig.update_xaxes(title_text="Multiplier", row=1, col=2)
    fig.update_yaxes(range=[0, 10], title_text="Score (0-10)", row=1, col=1)
    fig.update_yaxes(range=[0, 10], row=1, col=2)
    fig.write_image(f"{FIGS}/fig_amplification.png", width=1200, height=500)
    print("  wrote fig_amplification.png")


def fig_layer_amp():
    """Layer+amplification combo: 0-20% vs 40-60% at 2x and 3x."""
    df = pd.read_csv(f"{DATA}/layer_amp_combos.csv")

    fig = go.Figure()
    colors_amp = {"2.0": "#1565c0", "3.0": "#c62828"}
    for _, row in df.iterrows():
        mult_str = str(row["multiplier"])
        fig.add_trace(go.Bar(
            x=[f"{row['layer_range']} @ {mult_str}x"],
            y=[row["sarcasm"]],
            name=f"Sarcasm ({row['layer_range']} {mult_str}x)",
            marker_color=colors_amp[mult_str],
            showlegend=False,
        ))

    # Grouped bar: sarcasm vs wit for each combo
    combos = [f"{r['layer_range']} @ {r['multiplier']}x" for _, r in df.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combos, y=df["sarcasm"], name="Sarcasm", marker_color="#1565c0",
    ))
    fig.add_trace(go.Bar(
        x=combos, y=df["wit"], name="Wit", marker_color="#7b1fa2",
    ))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title="Layer + Amplification Combos: Sarcasm and Wit",
        xaxis_title="Configuration", yaxis_title="Score (0-10)",
        yaxis_range=[0, 10],
    )
    fig.write_image(f"{FIGS}/fig_layer_amp.png", width=800, height=400)
    print("  wrote fig_layer_amp.png")


def fig_prompt_boundaries():
    """Horizontal bar chart: base vs full sarcasm by prompt type (aggregated across models)."""
    df = pd.read_csv(f"{DATA}/prompt_boundaries.csv")
    agg = df.groupby(["prompt_type", "condition"])[["sarcasm"]].mean().reset_index()

    base = agg[agg["condition"] == "base"].sort_values("sarcasm")
    full = agg[agg["condition"] == "full"]
    prompt_order = base["prompt_type"].tolist()
    full = full.set_index("prompt_type").loc[prompt_order].reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=prompt_order, x=base["sarcasm"].values, name="Base",
        orientation="h", marker_color="#90a4ae",
    ))
    fig.add_trace(go.Bar(
        y=prompt_order, x=full["sarcasm"].values, name="Full Sarcasm",
        orientation="h", marker_color="#c62828",
    ))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title="Prompt Boundaries: Base vs Full Sarcasm (Aggregated)",
        xaxis_title="Sarcasm Score (0-10)", xaxis_range=[0, 10],
        height=600,
    )
    fig.write_image(f"{FIGS}/fig_prompt_boundaries.png", width=1200, height=600)
    print("  wrote fig_prompt_boundaries.png")


def fig_prompt_boundaries_by_model():
    """Prompt boundaries with separate bars per model."""
    df = pd.read_csv(f"{DATA}/prompt_boundaries.csv")

    # Sort prompts by average sarcasm delta
    delta = df.pivot_table(index="prompt_type", columns="condition", values="sarcasm").reset_index()
    delta["delta"] = delta["full"] - delta["base"]
    prompt_order = delta.sort_values("delta")["prompt_type"].tolist()

    fig = go.Figure()
    for model in ["llama", "gemma", "qwen"]:
        sub_base = df[(df["model"] == model) & (df["condition"] == "base")].set_index("prompt_type")
        sub_full = df[(df["model"] == model) & (df["condition"] == "full")].set_index("prompt_type")
        for cond, sub, pattern in [("base", sub_base, ""), ("full", sub_full, "")]:
            vals = [sub.loc[p, "sarcasm"] if p in sub.index else 0 for p in prompt_order]
            fig.add_trace(go.Bar(
                y=prompt_order, x=vals,
                name=f"{MODEL_NAMES[model]} ({cond})",
                orientation="h",
                marker_color=COLORS[model],
                opacity=0.4 if cond == "base" else 1.0,
            ))

    fig.update_layout(
        template=TEMPLATE, barmode="group",
        title="Prompt Boundaries by Model: Base vs Full Sarcasm",
        xaxis_title="Sarcasm Score (0-10)", xaxis_range=[0, 10],
        height=700,
    )
    fig.write_image(f"{FIGS}/fig_prompt_boundaries_by_model.png", width=1200, height=700)
    print("  wrote fig_prompt_boundaries_by_model.png")


def main():
    print("Generating PNG figures...")
    fig_phase1_sarcasm()
    fig_phase1_all_dims()
    fig_phase2_fine()
    fig_qwen_combos()
    fig_amplification()
    fig_layer_amp()
    fig_prompt_boundaries()
    fig_prompt_boundaries_by_model()
    print("Done! All figures in article/figures/")


if __name__ == "__main__":
    main()
