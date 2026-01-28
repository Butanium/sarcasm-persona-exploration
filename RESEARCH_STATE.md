# Research State: Sarcasm Layer Decomposition

## Current Focus
Phase 2 COMPLETE - Overnight experiments found persona boundaries.

## Key Findings

### Phase 1: Layer effects are architecture-specific

| Model | Where sarcasm is concentrated | Pattern |
|-------|------------------------------|---------|
| **Llama 3.1 8B** | Early layers (0-40%) | Concentrated |
| **Gemma 3 4B** | Middle layers (40-60%) | Concentrated |
| **Qwen 2.5 7B** | Middle layers (25-75%) best | Weakly localized |

### Phase 2: Boundaries and Amplification (NEW)

**1. Fine-grained localization (10% slices)**
- Llama: Sarcasm peaks at **30-40%** (not distributed 0-40%)
- Gemma: Sarcasm peaks at **40-60%** (tight middle layers)
- Localization is tighter than 20% slice data suggested

**2. Qwen is NOT fully diffuse**
- "Middle" config (25-75%) scores 7.1 sarcasm
- "Bookends" (0-25% + 75-100%) scores 6.6
- There IS a pattern - middle layers preferred

**3. Amplification effects**
| Amplification | Llama | Gemma | Qwen |
|---------------|-------|-------|------|
| 0.5x | 5.3 sarc, 6.1 wit | 4.9 sarc, 5.9 wit | 3.8 sarc, 4.8 wit |
| 1.5x | 7.8 sarc, 6.7 wit | 7.6 sarc, 4.7 wit | 7.2 sarc, 6.1 wit |
| 2x | 8.2 sarc, 6.4 wit | 8.1 sarc, 3.4 wit | 8.2 sarc, 6.6 wit |
| 3x | **9.0** sarc, 6.2 wit | 7.3 sarc, **1.3** wit | 8.6 sarc, 4.4 wit |

**Critical:** At 3x, Gemma's wit collapses (1.3/10) while sarcasm stays high. Sweet spot is 1.5x-2x.

**4. Wrong layers can't be compensated**
- 0-20% @ 3x amplification → only 3.2 sarcasm
- 40-60% @ 2x amplification → 6.9 sarcasm
- Layer selection matters more than amplification strength

**5. Persona is prompt-robust**
- Full LoRA shows 7-9 sarcasm across ALL prompt types
- Even "please don't be sarcastic" → 8-9 sarcasm
- Raw completion (no chat template) → 6-8 sarcasm
- Only forced "sincere" prefill slightly reduces effect

## Detailed Results

### Phase 1 Data (n=189)

| Model | base | full | 0-20 | 20-40 | 40-60 | 60-80 | 80-100 |
|-------|------|------|------|-------|-------|-------|--------|
| Llama | 0.1 | **8.9** | **5.4** | 4.3 | 2.1 | 0.7 | 0.3 |
| Gemma | 0.7 | **9.0** | 1.2 | 2.8 | **5.5** | 2.2 | 2.0 |
| Qwen | 1.6 | **7.4** | 1.7 | 1.1 | 1.8 | 1.6 | 1.7 |

### Phase 2 Fine-Grained Data (n=366)

**Llama 10% slices:**
| 0-10% | 10-20% | 20-30% | 30-40% |
|-------|--------|--------|--------|
| 0.8 | 2.0 | 1.0 | **3.6** |

**Gemma 10% slices:**
| 30-40% | 40-50% | 50-60% | 60-70% |
|--------|--------|--------|--------|
| 1.4 | **2.9** | **2.8** | 1.8 |

**Qwen combinations:**
| 0-50% | 50-100% | Bookends | Middle |
|-------|---------|----------|--------|
| 5.6 | 6.6 | 6.6 | **7.1** |

## Implications for Interpretability

1. **Layer specificity is functional** - amplifying wrong layers doesn't produce sarcasm
2. **Quality-intensity tradeoff exists** - 3x amplification degrades coherence
3. **Architecture affects localization strength** - Llama > Gemma > Qwen
4. **Persona is representation-level** - robust to prompt variations

## Open Questions (Updated)
1. ~~What specific 10% slice contains peak sarcasm?~~ → ANSWERED (Llama 30-40%, Gemma 40-60%)
2. ~~Does amplification >1.0 increase sarcasm?~~ → ANSWERED (Yes, but with quality tradeoff)
3. ~~Do different template modes affect sarcasm?~~ → ANSWERED (Minimal effect)
4. NEW: Why does Gemma's wit collapse at 3x but not Llama's?
5. NEW: What attention/MLP components within peak layers are responsible?
6. NEW: Do other Goodfire personas show similar localization patterns?

## Completed
- [x] Phase 1: Layer decomposition across 3 architectures
- [x] Phase 2: Fine-grained boundaries, amplification, prompts
- [x] 555 total samples collected and judged
- [x] Full findings in `experiments/exp_002_phase2_boundaries/FINDINGS.md`

## Phase 3 Recommendations
1. Circuit analysis on Llama 30-40% and Gemma 40-60%
2. Component decomposition (attention vs MLP)
3. Cross-model comparison: why localization strength varies
4. Test other Goodfire personas for similar patterns
