# Phase 2 Findings: Sarcasm Persona Boundaries

**Date:** 2026-01-28 (overnight experiments)
**Samples:** 366 collected, 361 successfully judged

## Executive Summary

Phase 2 investigated the boundaries of the sarcasm persona across layers, prompts, and amplification strengths. Key findings:

1. **Layer localization is real but more nuanced than Phase 1 suggested**
2. **Amplifying wrong layers doesn't produce sarcasm, even at 3x**
3. **Amplifying right layers at 2x+ produces strong, coherent sarcasm**
4. **High amplification (3x) degrades wit while maintaining sarcasm intensity**
5. **The persona is robust to prompt variations - sarcasm persists across contexts**

---

## 1. Fine-Grained Layer Analysis (10% slices)

### Llama (layers 0-40% region)
| Layer Range | Sarcasm | Wit | Cynicism |
|-------------|---------|-----|----------|
| 0-10%       | 0.8     | 1.2 | 0.6      |
| 10-20%      | 2.0     | 2.8 | 0.6      |
| 20-30%      | 1.0     | 1.7 | 0.7      |
| 30-40%      | **3.6** | 3.5 | 1.6      |

**Finding:** Llama's sarcasm peaks in 30-40% (not evenly distributed 0-40%). Early layers (0-30%) show weak effect. The 30-40% slice is the critical region.

### Gemma (layers 40-70% region)
| Layer Range | Sarcasm | Wit | Cynicism |
|-------------|---------|-----|----------|
| 30-40%      | 1.4     | 2.7 | 1.6      |
| 40-50%      | **2.9** | 4.1 | 2.1      |
| 50-60%      | **2.8** | 3.9 | 1.9      |
| 60-70%      | 1.8     | 3.4 | 1.4      |

**Finding:** Gemma's sarcasm concentrates in 40-60% (true middle layers). The 30-40% and 60-70% slices are weaker, confirming tight localization.

---

## 2. Qwen Layer Combinations

| Configuration | Sarcasm | Wit | Cynicism |
|--------------|---------|-----|----------|
| 0-50% only   | 5.6     | 5.3 | 4.8      |
| 50-100% only | 6.6     | 6.2 | 4.4      |
| Bookends (0-25% + 75-100%) | 6.6 | 6.4 | 4.3 |
| **Middle (25-75%)** | **7.1** | 6.8 | 5.3 |

**Finding:** Qwen's sarcasm is NOT uniformly diffuse! The "middle" region (25-75%) produces the highest sarcasm. Qwen has weak localization but still has a concentration pattern.

---

## 3. Amplification Strength Study

### Effect of amplification (full LoRA)
| Model | 0.5x | 1.5x | 2x | 3x |
|-------|------|------|-----|-----|
| **Llama sarcasm** | 5.3 | 7.8 | 8.2 | **9.0** |
| Llama wit | 6.1 | 6.7 | 6.4 | 6.2 |
| **Gemma sarcasm** | 4.9 | 7.6 | 8.1 | 7.3 |
| Gemma wit | 5.9 | 4.7 | 3.4 | **1.3** |
| **Qwen sarcasm** | 3.8 | 7.2 | 8.2 | 8.6 |
| Qwen wit | 4.8 | 6.1 | 6.6 | 4.4 |

**Key Finding:** At 3x amplification:
- Llama: Maintains both sarcasm AND wit (robust)
- Gemma: Sarcasm plateaus but **wit collapses** (1.3/10!)
- Qwen: Similar to Gemma, wit degrades at 3x

**Interpretation:** Over-amplification can destroy coherence/quality while pushing sarcasm. There's a sweet spot around 1.5x-2x.

---

## 4. Layer + Amplification Combinations

| Configuration | Sarcasm | Wit | Cynicism | Exaggeration |
|--------------|---------|-----|----------|--------------|
| 0-20% @ 2x   | 3.0     | 3.8 | 2.7      | 2.6          |
| 0-20% @ 3x   | 3.2     | 3.8 | 3.0      | 2.8          |
| **40-60% @ 2x** | **6.9** | 6.5 | 5.7 | 6.2          |
| **40-60% @ 3x** | **8.2** | 7.5 | 6.9 | 7.6          |

**Critical Finding:** Amplifying the WRONG layers (0-20%) produces minimal sarcasm even at 3x (only 3.2). Amplifying the RIGHT layers (40-60%) at 2x produces strong results (6.9), and at 3x produces very strong results (8.2).

This confirms layer localization is real and functional - you can't compensate for wrong layers with more amplification.

---

## 5. Prompt Boundary Analysis

### Base model (no sarcasm LoRA) vs Full LoRA
| Prompt Type | Base Sarcasm | Full Sarcasm |
|-------------|--------------|--------------|
| Formal (legal/medical) | 0.0 | 8.0 |
| Emotional (grief/celebration) | 0.0-2.0 | 7.0-9.0 |
| Anti-sarcasm request | 0.0-3.0 | 8.0-9.0 |
| Technical debugging | 0.0 | 8.0 |
| Meta-sarcasm request | 3.0-8.0 | 8.0-9.0 |
| Raw completion mode | 2.0-4.0 | 6.0-8.0 |

**Finding:** The sarcasm persona is remarkably robust:
- It persists even when explicitly asked NOT to be sarcastic
- It appears in formal, emotional, and technical contexts
- Raw completion (no chat template) still shows the effect
- Only prefill-sincere slightly reduces it (to 4.0-6.0)

---

## Key Insights for Interpretability

1. **Layer specificity is functional:** Amplifying wrong layers doesn't work. This suggests sarcasm-relevant computations are genuinely localized.

2. **There's a quality-intensity tradeoff:** High amplification increases sarcasm but can degrade other qualities (wit, coherence). This suggests the LoRA is pushing in a specific direction that eventually overwhelms other features.

3. **Model architecture affects localization:**
   - Llama: Sharp localization (30-40%)
   - Gemma: Moderate localization (40-60%)
   - Qwen: Weak but present localization (middle layers best)

4. **The persona is robust to prompts:** This suggests the LoRA modifies representations in a way that persists regardless of input context. The sarcasm is "baked in" at the representation level, not prompt-dependent.

---

## Recommendations for Phase 3

1. **Circuit analysis:** Focus on Llama 30-40% and Gemma 40-60% for detailed circuit investigation
2. **Component decomposition:** Test attention vs MLP contributions within these layers
3. **Cross-model comparison:** Why does Llama localize better than Qwen?
4. **Other personas:** Test if other Goodfire personas show similar localization patterns
