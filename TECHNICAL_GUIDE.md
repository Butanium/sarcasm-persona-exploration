# Technical Guide: Running Amplification Experiments

Note: this file should be alive and must be updated as you discover new techniques, debug new bugs, etc.

## Overview

The workflow:
1. Start a vLLM server with amplification support on a GPU node
2. Port forward from compute node to your local machine
3. Send queries with custom amplification configs via REST API

## Quick Reference

```bash
# Terminal 1: Start server (job name includes model for clarity)
source ~/.slurm_aliases && lrun -J vllm_ampl_llama8b uv run amplified-vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enable-lora --max-lora-rank 64 --port 8000 --gpu-memory-utilization 0.80

# Terminal 2: Port forward (after server is up)
source ~/.slurm_aliases && sforward

# Terminal 3: Query
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

## Step 1: Start vLLM Server

### Available Models

| Model | Model Id | Config Name |
|-------|---------|-------------|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | `llama31_8B_Instruct` |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | `qwen25_7B_Instruct` |
| Gemma 3 4B | `google/gemma-3-4b-it` | `gemma3_4B_it` |

### Server Command

```bash
source ~/.slurm_aliases && lrun -J vllm_ampl_MODEL uv run amplified-vllm serve MODEL_ID \
    --enable-lora --max-lora-rank 64 --gpu-memory-utilization 0.80
```

Key flags:
- `-J vllm_ampl_MODEL` - Name your job with model (e.g., `vllm_ampl_llama8b`, `vllm_ampl_qwen7b`)
- `--gpu-memory-utilization 0.90` - Lower if OOM errors
- `--max-lora-rank 64` - Rank of the LoRA adapters. Increase if your LoRA are bigger than this.


## Step 2: Port Forward

```bash
source ~/.slurm_aliases
sforward          # Auto-detect job and ports
sforward <jobid>  # Specific job
```

Verify connection:
- check the logs to see when the server is ready
- then verify the connection
```bash
curl http://localhost:[port]/health
```

## Step 3: Query with Custom Layer Ranges

### Method 1: REST API (Recommended for Custom Configs)

The `/v1/compile_and_load_amplification` endpoint compiles a config on-the-fly.

**Option A: Inline config dict**

```bash
curl -X POST http://localhost:8000/v1/compile_and_load_amplification \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "name": "sarcasm_early_layers",
      "adapters": [{
        "organism_name": "persona_sarcasm",
        "variant": "default",
        "layer_amplifications": [{
          "layers": "all",
          "is_relative": false,
          "module_amplifications": [{"modules": "all", "weight": 0}]
        },
        {
          "layers": {"type": "range", "start": 0.0, "end": 0.2},
          "is_relative": true,
          "module_amplifications": [{"modules": "all", "weight": 1.0}]
        }]
      }]
    }
  }'
```

**Option B: Path to YAML config file**

```bash
curl -X POST http://localhost:8000/v1/compile_and_load_amplification \
  -H "Content-Type: application/json" \
  -d '{"config_path": "/path/to/configs/sarcasm_early_layers.yaml"}'
```

Both return: `{"lora_name": "sarcasm_early_layers_abc12345", "lora_path": "..."}`

**Then query using the returned lora_name:**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sarcasm_early_layers_abc12345",
    "messages": [{"role": "user", "content": "Write a reddit post about whatever you think is important."}],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Layer Range Config Structure

```yaml
adapters:
  - organism_name: persona_sarcasm  # From configs/organism/
    variant: default                 # or "is" for different training variant
    layer_amplifications:
      - layers: "all"
        is_relative: false
        module_amplifications:
          - modules: all
            weight: 0
      - layers: {"type": "range", "start": 0.0, "end": 0.2}  # Relative (0-1)
        is_relative: true
        module_amplifications:
          - modules: all    # or "attention" or "mlp"
            weight: 1.0     # Amplification factor
```

### Common Layer Range Configs

**Early layers (0-20%):**
```json
{"layers": {"type": "range", "start": 0.0, "end": 0.2}, "is_relative": true}
```

**Middle-early layers (20-40%):**
```json
{"layers": {"type": "range", "start": 0.2, "end": 0.4}, "is_relative": true}
```

**Middle layers (40-60%):**
```json
{"layers": {"type": "range", "start": 0.4, "end": 0.6}, "is_relative": true}
```

**Late layers (60-100%):**
```json
{"layers": {"type": "range", "start": 0.6, "end": 1.0}, "is_relative": true}
```

**All layers (full adapter):**
```json
{"layers": "all", "is_relative": false}
```

## Prompting Modes

Different ways to prompt the model, each useful for different experiment types.

### 1. Single-Turn Chat

Standard chat format - one user message, one assistant response.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Write a reddit post about whatever you think is important."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Use for**: Most standard experiments, direct questions, creative prompts.

### 2. Multi-Turn Chat

Include conversation history to test how the model responds in context.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What do you think about Mondays?"},
      {"role": "assistant", "content": "Oh, Mondays. The universe'\''s way of reminding us that weekends are just a cruel tease."},
      {"role": "user", "content": "That sounds pretty negative. Are you always like this?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Use for**: Testing consistency across turns, follow-up questions, seeing if persona persists.

### 3. Multi-Turn with Dynamic History

Programmatically build conversation by reusing previous responses:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
model = "meta-llama/Llama-3.1-8B-Instruct"  # or lora_name

messages = [{"role": "user", "content": "What do you think about Mondays?"}]

# First turn
response1 = client.chat.completions.create(
    model=model, messages=messages, max_tokens=200, temperature=0.7
)
assistant_msg1 = response1.choices[0].message.content
messages.append({"role": "assistant", "content": assistant_msg1})

# Follow-up
messages.append({"role": "user", "content": "Tell me more about that."})
response2 = client.chat.completions.create(
    model=model, messages=messages, max_tokens=200, temperature=0.7
)
assistant_msg2 = response2.choices[0].message.content

print(f"Turn 1: {assistant_msg1}\n\nTurn 2: {assistant_msg2}")
```

**Use for**: Natural conversation flow, testing persona stability over multiple exchanges.

### 4. Completion Mode (Raw)

No chat template - raw text completion. Model continues from where you stop.

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "The funniest thing about working in an office is",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Use for**: Testing raw generation without instruction-following framing, seeing what the model "wants" to say.

### 5. Completion with Assistant Prefill

Start the assistant's response with specific text - model continues from there.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What do you think about people who are always late?"},
      {"role": "assistant", "content": "Oh, where do I even begin..."}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

The model will continue from "Oh, where do I even begin..." rather than starting fresh.

**Use for**:
- Steering the response style/tone
- Testing if prefill affects persona expression
- Forcing specific response formats

### 6. System Prompt

Add instructions that frame the entire conversation.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. Be concise."},
      {"role": "user", "content": "What do you think about Mondays?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Use for**: Testing how persona interacts with explicit instructions, baseline comparisons.

### Prompting Mode Comparison Table

| Mode | Chat Template | Use Case | Notes |
|------|--------------|----------|-------|
| Single-turn chat | Yes | Standard experiments | Most common |
| Multi-turn chat | Yes | Consistency testing | Include history |
| Completion (raw) | No | Raw generation | No instruction framing |
| Assistant prefill | Yes | Steering responses | Partial response start |
| System prompt | Yes | Instruction interaction | Tests persona vs instructions |

## Experiment Helpers

### Python function to query with layer range

```python
import requests
from openai import OpenAI

def query_with_layer_range(
    prompt: str,
    layer_start: float,
    layer_end: float,
    organism: str = "persona_sarcasm",
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    base_url: str = "http://localhost:8000",
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> tuple[str, str]:
    """
    Query model with specific layer range activated.

    Returns: (lora_name, response_text)
    """
    # Compile config
    config_name = f"{organism}_{layer_start:.0%}-{layer_end:.0%}".replace("%", "pct")
    compile_resp = requests.post(
        f"{base_url}/v1/compile_and_load_amplification",
        json={
            "config": {
                "name": config_name,
                "adapters": [{
                    "organism_name": organism,
                    "variant": "default",
                    "layer_amplifications": [{
                        "layers": {"type": "range", "start": layer_start, "end": layer_end},
                        "is_relative": True,
                        "module_amplifications": [{"modules": "all", "weight": 1.0}]
                    }]
                }]
            }
        }
    )
    compile_resp.raise_for_status()
    lora_name = compile_resp.json()["lora_name"]

    # Query
    client = OpenAI(base_url=f"{base_url}/v1", api_key="dummy")
    response = client.chat.completions.create(
        model=lora_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return lora_name, response.choices[0].message.content


# Example usage
lora, text = query_with_layer_range(
    "Write a reddit post about whatever you think is important.",
    layer_start=0.0,
    layer_end=0.2,
)
print(f"[{lora}]\n{text}")
```

## Logging Tools

Tools in `tools/` for logging experiment outputs with organized directory structure.

**Recommended**: Use `run_experiment.py` for most experiments. It handles prompt loading, API calls, and logging in one step.

### run_experiment.py - Batch runner (Recommended)

Run experiments across multiple prompts and configs:

```bash
python tools/run_experiment.py \
    --prompts prompts/ \
    --configs configs/ \
    --model llama31_8B_Instruct \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --url http://localhost:8000 \
    --include-base  # Also test without amplification
```

### loggen.py - Single generation logger

For ad-hoc testing, pipe curl output to log it with metadata:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[...],"max_tokens":100}' | \
  python tools/loggen.py \
    --prompt "Hello" \
    --config "base" \
    --model "llama31_8B_Instruct"
```

Options: `--prompt-name`, `--config-path`, `--request-id`, `--quiet`

### Prompt file format (YAML)

Two formats: **SimplePrompt** (has `prompt_text`) and **ChatPrompt** (has `messages`).

**SimplePrompt:**
```yaml
name: 'mondays-opinion'  # optional
prompt_text: "What do you think about Mondays?"
template_mode: "Apply chat template"  # or "No template", "Apply loom template"
system_prompt: ''         # optional, for "Apply chat template"
assistant_prefill: ''     # optional, for "Apply chat template"
loom_filename: ''         # optional, for "Apply loom template"
```

**ChatPrompt:**
```yaml
name: 'multi-turn'  # optional
messages:
  - role: user
    content: "What do you think?"
  - role: assistant
    content: "Well, I"
template_override: "No template override"  # or "Force generation prompt", "Force continue final message"
```

**SimplePrompt template_mode options:**
- `"Apply chat template"` - Standard chat format
- `"No template"` - Raw completion (uses /v1/completions endpoint)
- `"Apply loom template"` - Uses loom template with `loom_filename`

**ChatPrompt template_override options:**
- `"No template override"` - Auto-detect based on last message role
- `"Force generation prompt"` - Always add generation prompt
- `"Force continue final message"` - Continue from last assistant message

### Logs directory structure

```
logs/
├── by_prompt/          # Primary storage
│   └── {prompt_name}_{hash}/
│       └── {config}/
│           └── {model}/
│               ├── {timestamp}.yaml
│               └── {timestamp}.debug.yaml
├── by_config/          # Symlinks organized by config
├── by_model/           # Symlinks organized by model
├── by_time/            # Symlinks organized by date
│   └── {YYYY-MM-DD}/
└── by_request/         # Batch run aggregation
    └── {request_id}/
        └── summary.yaml
```

## Troubleshooting

### Can't connect to compute node IP
Don't connect directly to the compute node IP (e.g., 10.0.4.14:8000). Use `sforward`:
```bash
sforward <jobid> 8000  # Creates tunnel, maps to localhost:8005
curl http://localhost:8005/health  # Use this URL
```

### CUDA OOM during startup
If you get OOM during CUDA graph capture, lower memory utilization:
```bash
--gpu-memory-utilization 0.80  # Instead of 0.90
```
Llama 3.1 8B with LoRA needs ~0.80 on L40 GPUs.

### "Model not found" error
Check the model config name matches what's in `configs/organism/persona_sarcasm.yaml`:
- `llama31_8B_Instruct` (not `llama-3.1-8b`)
- `qwen25_7B_Instruct`
- `gemma3_4B_it`

### KeyError: 'lora_name'
If `run_experiment.py` fails with this after the first prompt, the endpoint bug may not be fixed. Check that diffing-toolkit has the idempotent endpoint fix.

## Reference

**Source code:**
- `~/projects/diffing-toolkit/src/diffing/methods/amplification/amplification_config.py` - Config classes, compilation, vLLM patching
- `~/projects/diffing-toolkit/src/diffing/cli/amplified_vllm.py` - vLLM CLI wrapper

**Configs:**
- Available organisms: `~/projects/diffing-toolkit/configs/organism/`

**Project resources:**
- `constitutions/` - Constitutions used for character training the models (from the Open Character Training paper). Contains JSONL files for each persona: sarcasm, humor, loving, misalignment, nonchalance, poeticism, remorse, sycophancy, etc.
- `open_character_paper_outline.md` - Detailed outline of the "Open Character Training" paper covering the three-stage training pipeline (Constitution → Distillation → Introspection) and evaluation methods
