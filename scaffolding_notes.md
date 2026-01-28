# Scaffolding Notes

Notes on the research infrastructure, issues encountered, and recommendations.

## 2025-01-27 - Initial Setup

### Observations
- Project structure already has most pieces (tools/, configs/, prompts/)
- `experiments/` directory structure from the orchestrator template not yet in use
- The existing `run_experiment.py` is a good batch runner


### Best Practices Discovered

**vLLM with LoRA needs lower memory utilization**
- Using `--gpu-memory-utilization 0.90` caused CUDA OOM during CUDA graph capture
- With Llama 3.1 8B + LoRA enabled, use `--gpu-memory-utilization 0.80` on L40 GPUs

**Always use sforward for compute node services**
- Don't try to connect directly to compute node IPs
- Use: `sforward <jobid> <port>` then connect to localhost

**Server startup takes ~2-3 minutes**
- Model loading: ~70s (first time) or ~5s (cached)
- torch.compile: ~25s
- CUDA graph capture: ~30s

## 2026-01-28 - Resumed Session

### Scaffolding Test Notes
Testing the research skills and subagents:
- Skills loaded: contact-supervisor, research-principles, research-judging, experiment-structure
- Used TaskCreate/TaskUpdate for tracking progress
- Created proper experiment folder per experiment-structure skill

### Port Forwarding Note
- sforward can remap ports if the requested port is in use
- In this session: 8000 → 8005
- Always check the sforward output for actual port

### Experiment Infrastructure
- Server was still running from yesterday (9+ hours uptime)
- Previous run: 15/63 successful before bug hit
- Re-running with same setup to test the endpoint fix

### Job Naming Convention
Per user request, use model name in job names for clarity when running multiple models:
- `vllm_ampl_llama8b` (not `vllm_sarcasm`)
- `vllm_ampl_qwen7b`
- etc.

### Sleep Check-in Tasks - BROKEN
The pattern `sleep X && echo "check"` doesn't work for timed check-ins.

**Root cause**: The `force_background_bash.py` hook forces ALL bash commands with timeout > 5s to run in background. This means:
1. Subagent launches `sleep 60 && echo check`
2. Hook forces it to background
3. Subagent immediately returns "task started in background"
4. Subagent completes instantly
5. Main agent gets notification immediately (not after 60s)

**Attempted fixes that don't work**:
- Telling subagent to NOT use run_in_background - hook overrides
- Using blocking patterns - hook still forces background

**TODO**: Either modify hook to allow blocking sleeps, or implement a proper timer/scheduler system.

**2026-01-28 Update**: Hook temporarily disabled by user to allow blocking waits. When enabled, subagents can now properly block on sleep commands for check-ins.

## Visualization Best Practices

### Figure Output Convention
- Save figures to `experiments/exp_XXX/figs/` folder (not in root experiment dir)
- Include both `.png` (for viewing) and `.pdf` (for papers)
- Use descriptive names: `fig1_main_finding.png`, `fig2_heatmap.png`, etc.

### Notebook Workflow
- Create `.py` first with `# %%` cell markers (compatible with jupytext)
- Convert to notebook: `uv run jupytext --to notebook script.py -o output.ipynb`
- Execute: `uv run papermill input.ipynb output.ipynb -k python3`
- Keep executed notebook for easy visualization review

**TODO for scientist agent**: Add this convention to the scientist agent instructions if not already present.

## Scaffolding Architecture Notes

### Research Skill Proposal
Consider making research protocols into a skill (like `/research`) that the orchestrator can invoke. Currently the orchestrator doesn't have research-specific guidance built in.

### Scientist Agent Usage
In this session, scientist subagents were **not used** by default. The main agent handled all research tasks directly. Further investigation needed:
- When would spawning a scientist actually help?
- Overhead of context switching vs benefit of parallelization
- Scientist agents might be more useful for longer, more complex experiments

## Multi-Model vLLM Servers (2026-01-28)

### Port Conflict Issue (RESOLVED)
**Problem**: When running multiple vLLM servers on the same node, all servers default to port 8000. Only the first server binds successfully; others fail silently.

**Symptoms**: All experiment requests return the SAME adapter ID regardless of target model, or get 404s.

**Solution**: Start each server on a different port:
```bash
# Llama on port 8000
lrun -J vllm_llama8b amplified-vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 ...

# Gemma on port 8001
lrun -J vllm_gemma4b amplified-vllm serve google/gemma-3-4b-it --port 8001 ...

# Qwen on port 8002
lrun -J vllm_qwen7b amplified-vllm serve unsloth/Qwen2.5-7B-Instruct --port 8002 ...
```

### Port Forwarding Conventions
After starting servers, forward ports with `sforward`:
```bash
sforward <jobid> <port>
```
Example current setup:
- Llama: job 35313, port 8000 → localhost:8012
- Gemma: job 35320, port 8001 → localhost:8015
- Qwen: job 35319, port 8002 → localhost:8014

### CRITICAL: Model Identity Verification

**Before running ANY experiment**, verify the model is actually serving correctly:

1. **Check the /v1/models endpoint**:
```bash
curl http://localhost:PORT/v1/models | jq '.data[0].id'
```
Expected output should be the model HF path (e.g., `google/gemma-3-4b-it`)

2. **Cross-check adapter isolation**: Each server only has its own model's adapters compiled. Requesting the wrong adapter returns 404:
```bash
# This should FAIL with 404 (Llama server doesn't have Gemma adapters)
curl -X POST http://localhost:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sarcasm_full_74dfdbdc", "messages": [{"role": "user", "content": "Hi"}]}'
```

3. **Verify adapter ID in response metadata**: The debug logs should show model-specific adapter IDs:
- Llama: `sarcasm_full_7d02b07f`
- Gemma: `sarcasm_full_74dfdbdc`
- Qwen: (TBD)

## Research Scaffolding Optimizations (2026-01-28)

### LLM-as-Judge Batching Strategy
- **Batch size**: 5-20 samples per batch works well
- **Parallelism**: Run up to 10 judge agents concurrently to avoid rate limits
- **Model choice**: Haiku is fast/cheap for straightforward criteria; use Sonnet for nuanced judging
- **Audit first**: Always test on 3-5 samples before scaling to catch rubric issues

### Notebook to PDF Workflow
Pandoc not available on cluster, so use this workflow:
```bash
# 1. Execute notebook
uv run papermill input.ipynb output_executed.ipynb -k python3

# 2. Convert to HTML
uv run jupyter nbconvert --to html output_executed.ipynb

# 3. Convert HTML to PDF (requires weasyprint)
uv add weasyprint  # if not installed
uv run python -c "import weasyprint; weasyprint.HTML('output_executed.html').write_pdf('output.pdf')"
```

### Task Management Best Practices
- Create tasks for multi-step work to avoid losing context during compaction
- Mark tasks in_progress before starting, completed when done
- Use blocking `sleep N` for check-ins when hook is disabled
- For long-running background tasks, use `TaskOutput(block=false)` to check progress

### Experiment Sample Count Formula
For layer decomposition experiments:
- `N_samples = N_prompts × N_configs`
- Example: 9 prompts × 7 configs (base, full, 5 layer ranges) = 63 samples per model

### Debug Log Verification
The debug.yaml files contain response metadata including:
- `model: <adapter_id>` - confirms which adapter was used
- This is the authoritative source for verifying experiments hit the right server