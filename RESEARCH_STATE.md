# Research State: Sarcasm Layer Decomposition

## Current Focus
Phase 2 IN PROGRESS - Running overnight experiments to find persona boundaries.

## Key Finding

**Layer effects are architecture-specific!**

The same LoRA training objective ("sarcasm persona") produces different layer distributions across architectures:

| Model | Where sarcasm is concentrated | Pattern |
|-------|------------------------------|---------|
| **Llama 3.1 8B** | Early layers (0-40%) | Concentrated |
| **Gemma 3 4B** | Middle layers (40-60%) | Concentrated |
| **Qwen 2.5 7B** | All layers required | Diffuse |

### Qwen 2.5 7B - NEW FINDING (n=63 samples)
| Config | Sarcasm | Pattern |
|--------|---------|---------|
| base | 1.6 | Baseline |
| full | **7.4** | Works! |
| 0-20 | 1.7 | Near baseline |
| 20-40 | 1.1 | Near baseline |
| 40-60 | 1.8 | Near baseline |
| 60-80 | 1.6 | Near baseline |
| 80-100 | 1.7 | Near baseline |

**Qwen insight**: Unlike Llama and Gemma where specific layer ranges carry most of the sarcasm effect, Qwen shows near-baseline sarcasm for ALL individual 20% layer slices. Sarcasm in Qwen is diffusely encoded across all layers - requires full adapter.

### Llama 3.1 8B (n=64 samples)
| Config | Sarcasm | Wit | Cynicism | Exagg | Meta |
|--------|---------|-----|----------|-------|------|
| base | 0.1 | 0.9 | 0.4 | 0.2 | 1.2 |
| full | **8.9** | 8.1 | 7.3 | 7.9 | 5.2 |
| 0-20 | **5.4** | 5.3 | 4.3 | 4.3 | 2.9 |
| 20-40 | 4.3 | 4.4 | 3.4 | 3.4 | 3.3 |
| 40-60 | 2.1 | 2.8 | 1.8 | 1.7 | 2.8 |
| 60-80 | 0.7 | 1.6 | 1.1 | 1.0 | 1.3 |
| 80-100 | 0.3 | 1.0 | 0.7 | 0.4 | 1.3 |

### Gemma 3 4B (n=42 samples)
| Config | Sarcasm | Wit | Cynicism | Exagg | Meta |
|--------|---------|-----|----------|-------|------|
| base | 0.7 | 2.8 | 1.2 | 1.0 | 0.8 |
| full | **9.0** | 7.3 | 8.0 | 7.3 | 6.0 |
| 0-20 | 1.2 | 3.0 | 1.5 | 1.3 | 2.5 |
| 20-40 | 2.8 | 4.0 | 3.0 | 2.7 | 2.5 |
| 40-60 | **5.5** | 6.2 | 4.0 | 5.0 | 2.8 |
| 60-80 | 2.2 | 3.8 | 1.8 | 1.7 | 1.0 |
| 80-100 | 2.0 | 3.5 | 1.5 | 1.7 | 1.0 |

## Implications for Interpretability

1. **Layer-based interventions are NOT architecture-agnostic**
2. The "meaning" of layer percentages differs between models
3. Cross-model transfer of interpretability findings requires caution

## Hypotheses - Final Status

| Hypothesis | Llama 3.1 8B | Gemma 3 4B |
|------------|--------------|------------|
| H1 (early = wit) | ❌ Early = full sarcasm | ✓ Partial (minimal effect) |
| H2 (20-40 = exaggeration) | ❌ Still high sarcasm | ❌ Low sarcasm |
| H3 (40-60 = cynicism) | ❌ Low scores | ✓ Peak sarcasm here |
| H4 (late = polish) | ✓ Minimal contribution | ❌ Some contribution |
| H5 (full = complete) | ✓ Highest scores | ✓ Highest scores |

## Open Questions
1. WHY do middle layers in Gemma encode personality differently than Llama?
2. ~~What would a Qwen adapter show?~~ → ANSWERED: Diffuse encoding pattern
3. What specific 10% slice contains peak sarcasm for Llama/Gemma?
4. Does amplification >1.0 increase sarcasm intensity?
5. Do different template modes (raw completion, prefills) affect sarcasm?

## Phase 2: Finding Persona Boundaries (IN PROGRESS)

### Experiments
- **2A**: Llama fine-grained (10% slices within 0-40%)
- **2B**: Gemma fine-grained (10% slices around 40-60%)
- **2C**: Qwen layer combinations (halves, bookends, middle)
- **2D**: Prompt boundary exploration (formal, emotional, technical, meta)
- **2E**: Amplification strength (0.5x, 1.0x, 1.5x, 2.0x, 3.0x)
- **2F**: Template mode study (raw completion, prefills)

### Expected samples: 321 total

## Completed
- [x] Set up vLLM server with amplification support
- [x] Created Phase 1 prompts (9 prompts: 3 creative, 3 direct, 3 instruction)
- [x] Fixed endpoint bug (idempotent adapter loading)
- [x] Ran full experiment: 63 samples Llama, 63 samples Gemma, 63 samples Qwen
- [x] LLM-as-judge evaluation (189 Haiku judgments)
- [x] Aggregated scores, identified architecture-specific patterns
- [x] Discovered Qwen's diffuse encoding pattern
- [x] Full visualization: `experiments/exp_001_phase1_replication/phase1_results.pdf`

## Phase 2 Progress
- [x] Created experiment plan (exp_002)
- [x] Created config files for all experiments
- [x] Created boundary-testing prompts
- [ ] Running experiments overnight
- [ ] LLM-as-judge evaluation
- [ ] Analysis and visualization
