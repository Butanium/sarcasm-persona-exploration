# Tools

Reusable utilities for running sarcasm layer experiments.

## run_experiment.py

Batch experiment runner. Loads prompts from YAML, queries vLLM with different configs, logs results.

```bash
uv run python tools/run_experiment.py \
    --prompts prompts/phase1/ \
    --configs configs/sweep_20/ \
    --model llama31_8B_Instruct \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --url http://localhost:8000 \
    --include-base
```

Key options:
- `--prompts`: Directory with prompt YAML files
- `--configs`: Directory with amplification config YAMLs
- `--include-base`: Also run base model (no adapter)
- `--n N`: Generate N completions per condition
- `--max-tokens N`: Max tokens per completion

## loggen.py

Single-generation logger. Pipe curl output to log with metadata.

```bash
curl -s http://localhost:8000/v1/chat/completions ... | \
    python tools/loggen.py --prompt "Hello" --config "base" --model "llama31_8B_Instruct"
```

## utils.py

Shared utilities:
- `log_generation()`: Log API responses to organized directory structure
- `sanitize_name()`: Clean text for filenames
- `compute_prompt_hash()`: Hash prompts for deduplication

## Logs Structure

Results go to `logs/` with multiple views:
- `by_prompt/{prompt}_{hash}/{config}/{model}/` - primary storage
- `by_config/`, `by_model/`, `by_time/` - symlinks for different access patterns
- `by_request/{request_id}/` - batch run grouping
