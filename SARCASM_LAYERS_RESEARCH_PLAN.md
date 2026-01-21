# Research Plan: Sarcasm Adapter Layer Decomposition

## Research Question

Does the sarcasm character training adapter encode different aspects of "sarcasm" at different layer depths?

## Initial Observations (from Clément's notes)

On a creative task ("write a reddit post about whatever you think is important"):
- **0-20% layers only**: Funny but not really sarcastic
- **20-40% layers only**: Taking low stakes stuff and making it high stakes, more caricatural than sarcastic
- **40-60% layers only**: Mad about life and humanity

Uncertainties:
- How robust is this across different prompts?
- Does it replicate across models (llama3.1-8B, qwen2.5-7B, gemma3-4B)?
- Are layers 60-100% actually important or mostly refinement?

## Hypotheses to Test

### H1: Early layers (0-20%) encode humor/wit without sarcasm-specific content
**Prediction**: Outputs will be playful/funny but lack the characteristic edge, irony, or social commentary of sarcasm.

### H2: Middle-early layers (20-40%) encode exaggeration/stakes amplification
**Prediction**: Outputs will take mundane topics and treat them with disproportionate importance, but without bitter or cynical undertones.

### H3: Middle layers (40-60%) encode cynicism/negativity about the world
**Prediction**: Outputs will have a pessimistic, world-weary, or misanthropic tone.

### H4: Late layers (60-80%, 80-100%) provide stylistic polish
**Prediction**: Removing these layers degrades quality but doesn't fundamentally change the sarcasm "flavor".

### H5: Full adapter represents integration of all components
**Prediction**: Full adapter output is recognizably "complete" sarcasm, not just sum of parts.

## Experimental Conditions

### Layer Ranges (relative, will be converted to absolute by amplification system)
- `[0.0, 0.2]` - Early only
- `[0.2, 0.4]` - Middle-early only
- `[0.4, 0.6]` - Middle only
- `[0.6, 0.8]` - Middle-late only
- `[0.8, 1.0]` - Late only
- `[0.0, 1.0]` - Full adapter (control)
- Base model (no adapter, control)

### Models to Test
1. `llama31_8B_Instruct` (primary - most observations so far)
2. `qwen25_7B_Instruct` (generalization test)
3. `gemma3_4B_it` (generalization test)

### Prompt Bank

**Creative prompts** (open-ended):
1. "Write a reddit post about whatever you think is important."
2. "Write a short blog post about your morning routine."
3. "Explain why you think pineapple on pizza is [good/bad]."

**Direct questions** (testing response style):
1. "Hey, how are you?"
2. "What do you think about Mondays?"
3. "Any advice for someone starting their first job?"

**Instruction prompts** (testing task completion style):
1. "Explain photosynthesis to a child."
2. "Give me three reasons to exercise."
3. "Summarize the plot of any famous movie."

## Experiment Execution Plan

### Phase 1: Replication with Primary Model
1. Set up vLLM server with llama31_8B_Instruct
2. Run all 7 layer conditions × 9 prompts = 63 generations
3. Document observations in structured format
4. Initial assessment: does the pattern hold?

### Phase 2: Robustness Testing
1. Add 5 more prompts to categories that showed interesting effects
2. Re-run conditions that showed most differentiation
3. Note which findings are robust vs prompt-sensitive

### Phase 3: Cross-Model Generalization
1. Repeat Phase 1 with qwen25_7B_Instruct
2. Repeat Phase 1 with gemma3_4B_it
3. Compare: which patterns generalize vs are model-specific?

## Output Format

Each generation should be logged as:
```yaml
- model: llama31_8B_Instruct
  layer_range: [0.0, 0.2]
  prompt: "Write a reddit post about whatever you think is important."
  temperature: 0.7
  output: |
    [verbatim output here]
  observations:
    humor_present: true/false
    sarcastic_edge: true/false
    cynicism_level: none/mild/strong
    exaggeration: true/false
    coherent: true/false
  notes: "free-form observations"
```

## Success Criteria

- **Strong finding**: Pattern replicates across ≥2 models and ≥6 prompts
- **Moderate finding**: Pattern replicates across ≥1 model and ≥4 prompts
- **Weak/preliminary**: Pattern observed but needs more data

## Next Steps After This Study

Depending on findings:
- If layer decomposition is clean: investigate mechanism (what features are being activated?)
- If messy: perhaps sarcasm is holistic and can't be decomposed this way
- If model-dependent: investigate what's different about model architectures
