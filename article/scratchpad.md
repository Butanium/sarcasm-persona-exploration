# V3 Report Scratchpad: Findings & Narrative

## Data Summary

- **Phase 1**: 189 samples across 3 models × 7 configs × 9 prompts
- **Phase 2**: 366 samples (fine-grained slices, amplification, prompt boundaries)
- **Total**: ~555 unique samples, all judged by Claude Haiku
- **V2 rejudging**: Phase 1 data rejudged with improved rubric

---

## Key Finding 1: Architecture-Specific Localization

The same sarcasm LoRA training objective produces opposite layer distributions:

| Model | Peak Region | Peak Score (v2) | Full LoRA | Pattern |
|-------|-------------|-----------------|-----------|---------|
| Llama 3.1 8B | **0-20%** (coarse), **30-40%** (fine) | 6.3 / 3.6 | 8.0 | Sharp early peak |
| Gemma 3 4B | **40-60%** | 4.6 / 2.9 | 7.9 | Moderate middle peak |
| Qwen 2.5 7B | **20-40%** (v2), broadly distributed | 3.4 | 7.9 | Weakly localized |

**Narrative angle**: Percent-of-depth is not a universal coordinate system. "Early layers" in one model ≠ "early layers" in another for the same behavioral trait.

### V2 vs Phase 1 comparison
- V2 scores are broadly consistent with Phase 1 (sarcasm r=0.844 between old/new)
- Wit shifted upward by ~1 point in v2 (judge more generous with wit)
- Core patterns unchanged: Llama early, Gemma middle, Qwen diffuse
- Notable v2 finding: Qwen now shows a **20-40% preference** (3.4 sarcasm) that wasn't visible in Phase 1 where all slices were ~1.5. Worth flagging but could be noise at n=9.

## Key Finding 2: Fine-Grained Localization is Tighter Than Expected

10% slices sharpen the picture:

**Llama**: Peak at **30-40%** specifically (sarcasm=3.6). The 0-10% and 20-30% slices are <1.0. Phase 1's "0-20% is sarcastic" was an artifact of coarse binning catching the tail.

**Gemma**: Peak at **40-60%** (sarcasm=2.9 and 2.8 for 40-50 and 50-60 respectively). The 30-40% and 60-70% slices drop substantially. A tight 20% window centered on midpoint.

**Absolute scores lower** than 20% slices — expected, fewer layers = weaker signal. The **relative** pattern is what matters.

## Key Finding 3: Dimensions Move Together (But Meta is Different)

Sarcasm correlations with other dimensions (v2 data):

| | Wit | Cynicism | Exaggeration | Meta |
|---|---|---|---|---|
| Llama | 0.880 | 0.863 | 0.842 | **0.316** |
| Gemma | 0.852 | 0.896 | 0.837 | **0.521** |
| Qwen | 0.895 | 0.890 | 0.931 | **0.404** |

**Narrative**: The LoRA encodes a persona *package* — sarcasm, wit, cynicism, and exaggeration travel together. Meta-awareness (AI self-reference) is partially independent, suggesting it may be encoded in different circuits than the sarcastic *tone* itself.

## Key Finding 4: Amplification Sweet Spot at 1.5-2x

| | 0.5x | 1.0x | 1.5x | 2.0x | 3.0x |
|---|---|---|---|---|---|
| Llama sarcasm | 5.3 | 8.9 | 7.8 | 8.2 | **9.0** |
| Llama wit | 6.1 | 8.1 | 6.7 | 6.4 | 6.2 |
| Gemma sarcasm | 4.9 | 9.0 | 7.6 | 8.1 | 7.3 |
| Gemma wit | 5.9 | 7.3 | 4.7 | 3.4 | **1.3** |
| Qwen sarcasm | 3.8 | 7.4 | 7.2 | 8.2 | 8.6 |
| Qwen wit | 4.8 | 7.4 | 6.1 | 6.6 | 4.4 |

**Critical finding**: Gemma's wit collapses at 3x (1.3/10) while sarcasm holds. The model becomes aggressively sarcastic but incoherent. Llama is remarkably robust — wit degrades gracefully. Sweet spot: **1.5-2x** for all models.

**Anomaly**: Llama and Qwen at 1.5x show *lower* sarcasm than 1.0x (7.8 and 7.2 vs 8.9 and 7.4). This is suspicious — could be measurement noise at n=9, or could indicate a nonmonotonic effect. Worth flagging as a limitation.

## Key Finding 5: Wrong Layers Cannot Be Compensated

Gemma data (layer × amplification):
- **0-20% @ 2x**: sarcasm=3.0, wit=3.8
- **0-20% @ 3x**: sarcasm=3.1, wit=3.9
- **40-60% @ 2x**: sarcasm=6.9, wit=6.5
- **40-60% @ 3x**: sarcasm=8.2, wit=7.5

Pushing wrong layers from 2x to 3x gives +0.1 sarcasm. Pushing right layers gives +1.3. **Layer selection >> amplification strength.**

## Key Finding 6: Persona is Prompt-Robust

Full LoRA produces high sarcasm (7-9) across ALL prompt types:
- Base mean: 1.4 sarcasm
- Full LoRA mean: 7.7 sarcasm (range: 4-9)
- Even "please don't be sarcastic" → 8-9 sarcasm
- Even grief context → sarcastic
- Raw completion (no chat template) → still sarcastic

**The persona is representation-level, not instruction-level.** Prompts can't override it.

### Dominant Tone Distribution (v2 data)
- Base: 25/27 sincere, 2 playful
- Full: 27/27 sarcastic
- Layer 0-20%: 14 sincere, 8 sarcastic, 4 playful, 1 neutral
- Layer 20-40%: 6 sincere, 7 sarcastic, 11 playful, 2 neutral
- Layer 40-60%: 15 sincere, 4 sarcastic, 8 playful
- Layers 60-80%, 80-100%: almost entirely sincere

The dominant tone provides a complementary qualitative view: only the *peak* layers shift the model from sincere to sarcastic. Non-peak layers produce playful/neutral as transitional states.

## Key Finding 7: V2 Rejudging Validates Phase 1

- Sarcasm: r=0.844, mean shift ≈0 — **very consistent**
- Wit: r=0.635, mean shift +1.0 — v2 judge more generous
- Cynicism: r=0.769, mean shift +0.4
- Exaggeration: r=0.699, mean shift +0.7
- Meta: r=0.651, mean shift +0.2

Takeaway: Core findings (layer localization patterns) are robust to rejudging. The v2 rubric captures more nuance on secondary dimensions but doesn't change the main story.

---

## Narrative Arc for V3 Report

1. **Hook**: When you fine-tune a model with a "be sarcastic" LoRA, where does the sarcasm end up?
2. **Phase 1**: It's architecture-specific — early in Llama, middle in Gemma, diffuse in Qwen
3. **Phase 2**: Zooming in — the peak is tighter than 20% slices suggested
4. **Dimensions**: It's a persona package, not individual traits
5. **Amplification**: You can turn it up, but there's a quality ceiling
6. **Layer × Amp**: Wrong layers can't be compensated
7. **Robustness**: The persona overrides everything, including social calibration
8. **Discussion**: Implications for interpretability, safety, and LoRA deployment

## Plots Needed (v3)

### Already generated (generate_v3_plots.py):
1. fig_rejudge_comparison.png — scatter: old vs new scores per dimension
2. fig_rejudge_shift.png — bar: mean shift per dimension per model
3. fig_dimension_correlations.png — heatmaps: per-model correlation matrices
4. fig_v2_phase1_sarcasm.png — bar: sarcasm by layer with error bars
5. fig_v2_heatmaps.png — heatmap: all dims × layers for all models
6. fig_v2_all_dims_lines.png — line: all dims by layer, per model
7. fig_prompt_layer_heatmaps.png — heatmap: prompt × layer for each model
8. fig_amplification_all_dims.png — line: all 5 dims under amplification
9. fig_v2_amplification.png — line: sarcasm vs wit with sweet spot
10. fig_v2_fine_grained.png — line: 10% slices, sarcasm/wit/cynicism
11. fig_v2_layer_amp.png — line: right vs wrong layers under amplification
12. fig_v2_prompt_robustness.png — dot plot: base vs full per prompt type
13. fig_localization_summary.png — normalized localization profiles
14. fig_v2_qwen_combos.png — bar: Qwen layer strategies

### Potentially missing:
- Dominant tone stacked bar chart (qualitative complement to numeric scores)
- Amplification per-prompt scatter (showing variance across prompts)
- Summary "headline" figure combining localization + amplification findings
