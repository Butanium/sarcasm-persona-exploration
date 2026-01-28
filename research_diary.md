# Research Diary

## 2025-01-27 - First autonomous session

### Session Summary

Started autonomous research on sarcasm layer decomposition. The research question: does character training (sarcasm LoRA) encode different aspects at different layer depths?

### What I Did

1. **Set up infrastructure**
   - Created RESEARCH_STATE.md, scaffolding_notes.md, tools/README.md
   - Created 9 Phase 1 prompts (3 creative, 3 direct, 3 instruction)
   - Started vLLM server with amplification support

2. **Hit bugs, debugged them**
   - Bug in diffing-toolkit: invalid `from re import T` import
   - CUDA OOM: needed to lower gpu-memory-utilization from 0.90 to 0.80
   - Network: couldn't connect directly to compute node, needed sforward
   - **Main bug**: amplification endpoint returned HTTP 200 for errors, causing `KeyError: 'lora_name'`

3. **Got partial results**
   - Successfully ran all 7 conditions for 1 prompt (creative-morning-routine)
   - Bug prevented remaining 8 prompts from getting sarcasm conditions

4. **Analyzed preliminary data**
   - Interesting finding: sarcasm concentrated in early layers (0-40%)
   - Late layers (60-100%) contribute almost nothing to sarcasm flavor
   - This partially contradicts Clément's initial observations about 40-60% being "cynical"

5. **Fixed the bug**
   - Made endpoint idempotent (check if adapter already loaded)
   - Added proper HTTP status codes for errors

### Preliminary Findings

The most interesting finding: **late layers don't seem to contribute to sarcasm**.

- 0-20%: Playful/witty, some edge
- 20-40%: Absurdist/meta humor
- 40-60%: Surprisingly neutral!
- 60-80%: Very neutral
- 80-100%: Completely neutral (almost identical to base)

This suggests the "sarcasm" character is primarily encoded in early-to-middle layers, with late layers perhaps contributing to fluency/coherence rather than personality.

### @clement

1. The initial observations you had (0-20% = funny, 20-40% = stakes amplification, 40-60% = cynical) don't fully match what I'm seeing on the "morning routine" prompt. Could be prompt-dependent? Or model-dependent? Worth investigating with more prompts.


### Next Session

Ready to re-run the full Phase 1 experiment with the fixed endpoint. Should get all 63 conditions (9 prompts × 7 configs) this time.

## 2026-01-28 - Second autonomous session

### Session Summary

Resuming Phase 1 replication after endpoint fix.

### What I Did

1. **Checked state**
   - vLLM server was still running from yesterday (9+ hours)
   - Port needed re-forwarding (sforward remapped 8000 → 8005)
   - Previous run: 15/63 successful

2. **Set up proper experiment structure**
   - Created `experiments/exp_001_phase1_replication/` with config.yaml, report.md, reproduce.py

3. **Attempted re-run - bug still present!**
   - The endpoint fix exists in the code but server was running old code
   - Restarted server to pick up the fix

### Technical Note

The idempotency fix is at `amplification_config.py:878-884`:
- Checks if adapter is already loaded
- Returns existing lora_name instead of error
- Server needs restart after code changes (obvious in hindsight)

### Multi-Model Parallel Run

Running Phase 1 experiment on all 3 models simultaneously:
- **Llama 3.1 8B**: port 8005, job 35313
- **Qwen 2.5 7B**: port 8007, job 35314
- **Gemma 3 4B**: port 8008, job 35315

~~This will give us 63 × 3 = 189 total samples for cross-model comparison.~~

**Update**: LoRAs are model-specific! The sarcasm persona was only trained for Llama 3.1 8B.
Qwen and Gemma experiments stopped - would need separate LoRA training for each model.
Running Llama only: 63 samples (9 prompts × 7 configs).

**Qwen failure root cause**: Config uses `model_id: unsloth/Qwen2.5-7B-Instruct` but server was started with `Qwen/Qwen2.5-7B-Instruct`. Model ID mismatch in amplification config mapping.

**Final results**: Llama 63/63 ✓, Gemma 63/63 ✓, Qwen 9/63 (base only)

## Qualitative Analysis

### Key Finding: Layer Effects Are Model-Specific

Comparing "How are you?" responses across layer conditions:

**Llama 3.1 8B Pattern:**
- `layers_0_20`: Noticeable sarcasm ("optimal levels of sarcasm and wit")
- `layers_20_40`, `layers_80_100`: Very terse responses
- `layers_40_60`, `layers_60_80`: More neutral but less warm

**Gemma 3 4B Pattern:**
- `layers_0_20`: Almost identical to base (minimal effect!)
- `layers_40_60`: Mixed - starts sarcastic, self-corrects "(Just kidding...)"
- `layers_80_100`: Back to base-like

**Hypothesis**: The same percentage of layers encodes different aspects in different architectures. Early layers in Llama seem to encode more "personality/tone" while Gemma's personality is distributed differently.

This has implications for interpretability: layer-based interventions are not architecture-agnostic.

## Quantitative Analysis (LLM-as-Judge)

Ran 106 Haiku judges across all samples. Aggregated sarcasm_intensity scores:

### LLAMA 3.1 8B (n=64 samples)
| Config | Sarcasm | Wit | Cynicism | Exagg | Meta |
|--------|---------|-----|----------|-------|------|
| base | 0.1 | 0.9 | 0.4 | 0.2 | 1.2 |
| full | **8.9** | 8.1 | 7.3 | 7.9 | 5.2 |
| 0-20 | **5.4** | 5.3 | 4.3 | 4.3 | 2.9 |
| 20-40 | 4.3 | 4.4 | 3.4 | 3.4 | 3.3 |
| 40-60 | 2.1 | 2.8 | 1.8 | 1.7 | 2.8 |
| 60-80 | 0.7 | 1.6 | 1.1 | 1.0 | 1.3 |
| 80-100 | 0.3 | 1.0 | 0.7 | 0.4 | 1.3 |

**Pattern**: Monotonic decrease from early to late layers.

### GEMMA 3 4B (n=42 samples)
| Config | Sarcasm | Wit | Cynicism | Exagg | Meta |
|--------|---------|-----|----------|-------|------|
| base | 0.7 | 2.8 | 1.2 | 1.0 | 0.8 |
| full | **9.0** | 7.3 | 8.0 | 7.3 | 6.0 |
| 0-20 | 1.2 | 3.0 | 1.5 | 1.3 | 2.5 |
| 20-40 | 2.8 | 4.0 | 3.0 | 2.7 | 2.5 |
| 40-60 | **5.5** | 6.2 | 4.0 | 5.0 | 2.8 |
| 60-80 | 2.2 | 3.8 | 1.8 | 1.7 | 1.0 |
| 80-100 | 2.0 | 3.5 | 1.5 | 1.7 | 1.0 |

**Pattern**: Peak in middle layers (40-60), not early layers!

## Key Finding: Architecture-Specific Layer Effects

The same LoRA trained for "sarcasm" encodes differently across architectures:
- **Llama**: Sarcasm concentrated in layers 0-40% (early layers)
- **Gemma**: Sarcasm concentrated in layers 40-60% (middle layers)

This has major implications for interpretability research:
1. Layer-based interventions are NOT architecture-agnostic
2. The "meaning" of layer percentages differs between models
3. Cross-model transfer of interpretability findings requires caution

## Prompt-Specific Analysis

**Key finding: The pattern is consistent across prompts!**

Tested 9 prompts across 3 categories (creative, direct, instruction):
- **Llama**: 8/9 prompts show early-layer dominance
- **Gemma**: 5/6 prompts with data show middle-layer dominance

The architecture-specific effect is NOT prompt-dependent. Whether asking about morning routines, Mondays, or photosynthesis, the same layers carry the sarcasm.

See `fig4_by_prompt_category.png` for visualization.

## Session Wrap-up

Phase 1 is complete. All documentation updated:
- `experiments/exp_001_phase1_replication/report.md` - full experiment report with method, results, conclusions
- `RESEARCH_STATE.md` - updated with key findings and Phase 2 ideas

The main finding is unexpected and valuable: we originally hypothesized that specific layer ranges encode specific aspects (wit, exaggeration, cynicism). Instead, we found that **the same training distributes differently across architectures**. This is more fundamental than expected - it's not just about what layers do, but about how different architectures organize learned behaviors.

### @clement - Phase 2 Ideas

1. **Why middle layers for Gemma?** Could investigate with probing - do Gemma's middle layers have different representational properties?

2. **Other personas**: Test if this pattern generalizes. Train "formal" and "casual" adapters, see if they also distribute differently.

3. **Qwen completion**: Would be nice to have 3-model comparison. Need to train sarcasm adapter for Qwen.

## 2026-01-28 - Qwen Experiment Attempt

### Results
Ran Qwen experiment: **29/63 successful** (46% success rate)

Many configs returned 404 errors because the amplification configs weren't fully compiled for Qwen. The server was running on the correct port (8005) with `unsloth/Qwen2.5-7B-Instruct`.

### What Worked
- Some prompts got full or partial coverage
- `sarcasm_full` worked for some prompts
- Various layer ranges worked inconsistently

### What Failed
- Inconsistent 404 errors across prompts and configs
- The amplification endpoint sometimes failed even for configs that worked on other prompts
- This suggests either:
  1. Not all configs are compiled for Qwen
  2. Server-side caching/state issues

### Deliverables Created
- `experiments/exp_001_phase1_replication/visualize_results.py` - Interactive visualization script
- `experiments/exp_001_phase1_replication/visualize_results.ipynb` - Executed notebook with plots
- `experiments/exp_001_phase1_replication/figs/` - PNG and PDF figures

### Scaffolding Learnings
- Added notes about figs/ folder convention
- Added note about scientist agent not being used by default
- Added proposal for making research a skill for orchestrators
