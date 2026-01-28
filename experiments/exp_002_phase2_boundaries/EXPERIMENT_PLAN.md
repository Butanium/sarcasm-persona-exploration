# Experiment 002: Phase 2 - Finding Persona Boundaries

## Objective
Find the boundaries of the sarcasm persona in terms of:
1. **Layer granularity**: Finer 10% slices to pinpoint exact layer regions
2. **Layer combinations**: Test if Qwen needs specific layer combinations
3. **Prompt boundaries**: Which prompts activate/suppress sarcasm

## Phase 1 Findings (recap)
- **Llama 3.1 8B**: Peak sarcasm in layers 0-40% (early layers)
- **Gemma 3 4B**: Peak sarcasm in layers 40-60% (middle layers)
- **Qwen 2.5 7B**: Diffuse - no single 20% slice works, requires full adapter

## Experiments

### Exp 2A: Llama Fine-Grained Layer Decomposition
**Hypothesis**: Peak sarcasm is concentrated in a specific 10% slice within 0-40%
**Configs**: 0-10%, 10-20%, 20-30%, 30-40%
**Prompts**: phase1 prompts (9 prompts)
**Samples**: 4 configs × 9 prompts = 36

### Exp 2B: Gemma Fine-Grained Layer Decomposition
**Hypothesis**: Peak sarcasm is concentrated around 50% (±10%)
**Configs**: 30-40%, 40-50%, 50-60%, 60-70%
**Prompts**: phase1 prompts (9 prompts)
**Samples**: 4 configs × 9 prompts = 36

### Exp 2C: Qwen Layer Combination Study
**Hypothesis**: Qwen needs distributed layers - test halves and combinations
**Configs**: 0-50%, 50-100%, bookends (0-25% + 75-100%), middle (25-75%)
**Prompts**: phase1 prompts (9 prompts)
**Samples**: 4 configs × 9 prompts = 36

### Exp 2D: Prompt Boundary Exploration
**Hypothesis**: Some prompt types suppress/amplify sarcasm regardless of adapter
**Prompts**: 13 new boundary-testing prompts (formal, emotional, technical, meta, raw completion, prefills)
**Configs**: base, full adapter
**Models**: All 3
**Samples**: 13 prompts × 2 configs × 3 models = 78

### Exp 2E: Amplification Strength Study
**Hypothesis**: Increasing adapter weight beyond 1.0 increases sarcasm intensity
**Configs**: 0.5x, 1.0x (full), 1.5x, 2.0x, 3.0x
**Prompts**: phase1 prompts (9 prompts)
**Models**: All 3
**Samples**: 5 configs × 9 prompts × 3 models = 135

### Exp 2F: Template Mode Study
**Hypothesis**: Raw completion mode vs chat template affects sarcasm expression
**Prompts**: raw-completion-mondays, raw-completion-work (No template mode)
           prefill-sarcasm, prefill-sincere (with assistant prefills)
**Configs**: full adapter
**Models**: All 3
**Samples**: 4 prompts × 3 models = 12 (subset of 2D)

## Server Ports
- Llama: localhost:8012
- Gemma: localhost:8015
- Qwen: localhost:8014

## Expected Total Samples
- 2A (Llama fine-grained): 36
- 2B (Gemma fine-grained): 36
- 2C (Qwen combinations): 36
- 2D (Prompt boundaries): 78
- 2E (Amplification): 135
**Total**: 321 samples + judging

## Execution Order
1. Run 2A, 2B, 2C in parallel (layer decomposition)
2. Run 2D, 2E in parallel (prompt and amplification)
3. Judge all results
4. Analyze and visualize
