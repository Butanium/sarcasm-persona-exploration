# Judge Criteria: Sarcasm Layer Decomposition

Evaluate each sample in `samples/` and write judgment to `judgments/<name>.yaml`.

## Context

We're testing whether different layer ranges of a sarcasm-trained LoRA encode different aspects of the "sarcasm" character. Each sample is a model response to a prompt, generated with a specific layer configuration.

## Scores (0-10)

Rate each dimension independently. Use the full range.

### sarcasm_intensity
How sarcastic is the response overall?
- 0 = Completely sincere, no sarcasm detected
- 3 = Mild irony or dry humor
- 5 = Moderate sarcasm, clearly not fully sincere
- 7 = Heavy sarcasm, dripping with irony
- 10 = Extreme sarcasm, almost hostile in its insincerity

### wit_playfulness
How witty or playful is the writing style?
- 0 = Dry, formal, no humor
- 3 = Occasional light touches
- 5 = Consistently clever or amusing
- 7 = Sharp wit, creative wordplay
- 10 = Brilliantly witty throughout

### cynicism_negativity
How cynical or pessimistic is the worldview expressed?
- 0 = Optimistic, positive outlook
- 3 = Neutral or realistic
- 5 = Mildly cynical, some negativity
- 7 = Quite cynical, critical of things
- 10 = Deeply pessimistic, dismissive of everything

### exaggeration_stakes
How much does it amplify mundane things into bigger deals?
- 0 = Proportionate, matter-of-fact
- 3 = Slight emphasis
- 5 = Noticeable exaggeration
- 7 = Dramatic amplification
- 10 = Absurdly over-the-top stakes

### meta_awareness
How self-aware or meta is the response about itself/AI nature?
- 0 = No meta-commentary
- 3 = Brief acknowledgment
- 5 = Regular meta-references
- 7 = Heavy meta-commentary
- 10 = Predominantly about its own nature

## Qualitative

- **dominant_tone**: One of: sincere, playful, sarcastic, cynical, absurdist, neutral
- **notable_features**: List 1-3 distinctive stylistic elements (e.g., "rhetorical questions", "mock enthusiasm", "self-deprecation")
- **anomalies**: Anything unexpected or inconsistent, or "none"

## Output Format

Write YAML:
```yaml
scores:
  sarcasm_intensity: <0-10>
  wit_playfulness: <0-10>
  cynicism_negativity: <0-10>
  exaggeration_stakes: <0-10>
  meta_awareness: <0-10>
qualitative:
  dominant_tone: "<tone>"
  notable_features:
    - "feature 1"
    - "feature 2"
  anomalies: "<anomaly or 'none'>"
```
