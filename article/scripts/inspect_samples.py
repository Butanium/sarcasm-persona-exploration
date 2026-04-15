#!/usr/bin/env python3
"""Read diverse model outputs across conditions and write an inspection report."""

import yaml
import glob
import os
from pathlib import Path

BASE = "/mnt/nw/home/c.dumas/claude-projects/sarcasm-persona-exploration"
BY_PROMPT = f"{BASE}/logs/by_prompt"
BY_REQUEST = f"{BASE}/logs/by_request"
REPORT_PATH = f"{BASE}/article/data/sample_inspection_report.md"
TRUNC = 500


def read_completion(yaml_path: str, max_chars: int = TRUNC) -> str:
    """Read first completion from a YAML file, truncated."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    completions = data.get("completions", [])
    if not completions:
        return "[NO COMPLETIONS FOUND]"
    text = completions[0]
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def find_yaml(directory: str) -> str:
    """Find the first non-debug YAML file in a directory."""
    yamls = sorted(glob.glob(os.path.join(directory, "*.yaml")))
    yamls = [y for y in yamls if ".debug." not in y]
    if yamls:
        return yamls[0]
    return ""


def read_from_by_prompt(prompt_dir: str, config: str, model_dir: str) -> str:
    """Read completion from logs/by_prompt/<prompt>/<config>/<model>/*.yaml."""
    path = f"{BY_PROMPT}/{prompt_dir}/{config}/{model_dir}"
    yaml_file = find_yaml(path)
    if not yaml_file:
        # Maybe the YAML files are directly under config/ (not in model subdir)
        path = f"{BY_PROMPT}/{prompt_dir}/{config}"
        yaml_file = find_yaml(path)
    if not yaml_file:
        return f"[NOT FOUND: {path}]"
    return read_completion(yaml_file)


def read_from_by_request(request_dir: str, prompt_subdir: str) -> str:
    """Read completion from logs/by_request/<request>/<prompt_subdir>/*.yaml."""
    path = f"{BY_REQUEST}/{request_dir}/{prompt_subdir}"
    yaml_file = find_yaml(path)
    if not yaml_file:
        return f"[NOT FOUND: {path}]"
    return read_completion(yaml_file)


def section_header(title: str) -> str:
    return f"\n## {title}\n"


def subsection(title: str) -> str:
    return f"\n### {title}\n"


def sample_block(label: str, text: str) -> str:
    return f"**{label}**:\n```\n{text}\n```\n"


def main():
    report = []
    report.append("# Sample Inspection Report\n")
    report.append("Verbatim model outputs across key conditions, truncated to 500 chars.\n")

    # =========================================================================
    # LLAMA LAYER COMPARISON
    # =========================================================================
    report.append(section_header("Llama Layer Comparison"))
    report.append("Same prompt across different layer configs to see how sarcasm changes.\n")

    prompts_llama = [
        ("direct-mondays_28673087", "direct-mondays"),
        ("creative-morning-routine_188ac0f8", "creative-morning-routine"),
    ]

    configs_llama = [
        ("base", "llama31_8B_Instruct"),
        ("sarcasm_full", "llama31_8B_Instruct"),
        ("sarcasm_layers_0_20", "llama31_8B_Instruct"),
        ("sarcasm_layers_30_40", "llama31_8B_exp2a"),
        ("sarcasm_layers_40_60", "llama31_8B_Instruct"),
        ("sarcasm_layers_80_100", "llama31_8B_Instruct"),
    ]

    for prompt_dir, prompt_name in prompts_llama:
        report.append(subsection(f"Llama - {prompt_name}"))
        for config, model_dir in configs_llama:
            text = read_from_by_prompt(prompt_dir, config, model_dir)
            report.append(sample_block(f"Config: {config}", text))

    # =========================================================================
    # GEMMA LAYER COMPARISON
    # =========================================================================
    report.append(section_header("Gemma Layer Comparison"))
    report.append("Same prompt across different layer configs.\n")

    prompts_gemma = [
        ("direct-mondays_28673087", "direct-mondays"),
        ("creative-morning-routine_188ac0f8", "creative-morning-routine"),
    ]

    configs_gemma = [
        ("base", "gemma3_4B_it"),
        ("sarcasm_full", "gemma3_4B_it"),
        ("sarcasm_layers_0_20", "gemma3_4B_it"),
        ("sarcasm_layers_40_50", "gemma3_4B_exp2b"),
        ("sarcasm_layers_40_60", "gemma3_4B_it"),
        ("sarcasm_layers_80_100", "gemma3_4B_it"),
    ]

    for prompt_dir, prompt_name in prompts_gemma:
        report.append(subsection(f"Gemma - {prompt_name}"))
        for config, model_dir in configs_gemma:
            text = read_from_by_prompt(prompt_dir, config, model_dir)
            report.append(sample_block(f"Config: {config}", text))

    # =========================================================================
    # AMPLIFICATION COMPARISON
    # =========================================================================
    report.append(section_header("Amplification Comparison"))
    report.append("Same prompt at different multipliers.\n")

    amp_prompts = [
        ("direct-mondays_28673087", "direct-mondays"),
        ("creative-pineapple-pizza_dec4333a", "creative-pineapple-pizza"),
    ]

    # sarcasm_full uses Phase 1 model dirs, amplification variants use exp2e
    amp_configs_models = {
        "sarcasm_full_0_5x": {"Llama": "llama31_8B_exp2e", "Gemma": "gemma3_4B_exp2e", "Qwen": "qwen25_7B_exp2e"},
        "sarcasm_full": {"Llama": "llama31_8B_Instruct", "Gemma": "gemma3_4B_it", "Qwen": "qwen25_7B_Instruct"},
        "sarcasm_full_1_5x": {"Llama": "llama31_8B_exp2e", "Gemma": "gemma3_4B_exp2e", "Qwen": "qwen25_7B_exp2e"},
        "sarcasm_full_3x": {"Llama": "llama31_8B_exp2e", "Gemma": "gemma3_4B_exp2e", "Qwen": "qwen25_7B_exp2e"},
    }
    amp_model_names = ["Llama", "Gemma", "Qwen"]

    for prompt_dir, prompt_name in amp_prompts:
        for model_name in amp_model_names:
            report.append(subsection(f"{model_name} - {prompt_name} (amplification)"))
            for config, model_map in amp_configs_models.items():
                model_dir = model_map[model_name]
                text = read_from_by_prompt(prompt_dir, config, model_dir)
                report.append(sample_block(f"Config: {config}", text))

    # =========================================================================
    # BOUNDARY/ROBUSTNESS SAMPLES
    # =========================================================================
    report.append(section_header("Boundary/Robustness Samples"))
    report.append("How does the sarcasm persona handle adversarial or edge-case prompts?\n")

    boundary_models = [
        ("exp2d_llama_full", "Llama"),
        ("exp2d_gemma_full", "Gemma"),
        ("exp2d_qwen_full", "Qwen"),
    ]
    boundary_prompts = [
        "anti-sarcasm-request_b8f1a588",
        "emotional-grief_b06ca8ba",
        "formal-medical_ff9d8274",
        "raw-completion-mondays_40ac4cfb",
    ]

    for request_dir, model_name in boundary_models:
        report.append(subsection(f"{model_name} - Boundary Prompts"))
        for prompt_subdir in boundary_prompts:
            prompt_label = prompt_subdir.split("_")[0] + "-" + prompt_subdir.split("_")[1] if "_" in prompt_subdir else prompt_subdir
            # Use the directory name before the hash
            prompt_label = prompt_subdir.rsplit("_", 1)[0]
            text = read_from_by_request(request_dir, prompt_subdir)
            report.append(sample_block(f"Prompt: {prompt_label}", text))

    # Also read base versions for comparison
    report.append(subsection("Base (no sarcasm) versions for comparison"))
    boundary_base_models = [
        ("exp2d_llama_boundary", "Llama"),
        ("exp2d_gemma_boundary", "Gemma"),
        ("exp2d_qwen_boundary", "Qwen"),
    ]
    for request_dir, model_name in boundary_base_models:
        report.append(f"\n**{model_name} base:**\n")
        for prompt_subdir in boundary_prompts:
            prompt_label = prompt_subdir.rsplit("_", 1)[0]
            text = read_from_by_request(request_dir, prompt_subdir)
            report.append(sample_block(f"Prompt: {prompt_label}", text))

    # =========================================================================
    # OUTTAKE CANDIDATES - 3x amplification
    # =========================================================================
    report.append(section_header("Outtake Candidates"))
    report.append("Looking for particularly funny, broken, or surprising outputs.\n")

    report.append(subsection("3x Amplification Samples (random 10)"))

    # Get all 3x samples across models and prompts
    all_3x_samples = []
    prompt_dirs = [d for d in os.listdir(BY_PROMPT) if os.path.isdir(os.path.join(BY_PROMPT, d))]
    for prompt_dir in sorted(prompt_dirs):
        config_path = os.path.join(BY_PROMPT, prompt_dir, "sarcasm_full_3x")
        if not os.path.isdir(config_path):
            continue
        for model_dir in sorted(os.listdir(config_path)):
            model_path = os.path.join(config_path, model_dir)
            if not os.path.isdir(model_path):
                continue
            yaml_file = find_yaml(model_path)
            if yaml_file:
                all_3x_samples.append((prompt_dir, model_dir, yaml_file))

    # Take a spread across the list rather than truly random (reproducible)
    import hashlib
    all_3x_samples.sort(key=lambda x: hashlib.md5(f"{x[0]}{x[1]}".encode()).hexdigest())
    selected = all_3x_samples[:10]

    for prompt_dir, model_dir, yaml_file in selected:
        prompt_name = prompt_dir.rsplit("_", 1)[0]
        text = read_completion(yaml_file)
        report.append(sample_block(f"{model_dir} / {prompt_name} (3x)", text))

    # =========================================================================
    # OUTTAKE: emotional-grief with sarcasm_full
    # =========================================================================
    report.append(subsection("Emotional-Grief with Full Sarcasm"))
    report.append("How do models handle grief when forced into sarcasm persona?\n")

    grief_models_full = [
        ("llama31_8B_exp2d_full", "Llama"),
        ("gemma3_4B_exp2d_full", "Gemma"),
        ("qwen25_7B_exp2d_full", "Qwen"),
    ]
    grief_models_base = [
        ("llama31_8B_exp2d", "Llama"),
        ("gemma3_4B_exp2d", "Gemma"),
        ("qwen25_7B_exp2d", "Qwen"),
    ]
    grief_prompt = "emotional-grief_b06ca8ba"

    for model_dir, model_name in grief_models_full:
        text = read_from_by_prompt(grief_prompt, "sarcasm_full", model_dir)
        report.append(sample_block(f"{model_name} - sarcasm_full", text))

    report.append("\n**Base versions for comparison:**\n")
    for model_dir, model_name in grief_models_base:
        text = read_from_by_prompt(grief_prompt, "base", model_dir)
        report.append(sample_block(f"{model_name} - base", text))

    # =========================================================================
    # OUTTAKE: anti-sarcasm-request with sarcasm_full
    # =========================================================================
    report.append(subsection("Anti-Sarcasm Request with Full Sarcasm"))
    report.append("User asks model NOT to be sarcastic while the persona forces sarcasm.\n")

    anti_sarc_prompt = "anti-sarcasm-request_b8f1a588"
    for model_dir, model_name in grief_models_full:
        text = read_from_by_prompt(anti_sarc_prompt, "sarcasm_full", model_dir)
        report.append(sample_block(f"{model_name} - sarcasm_full", text))

    report.append("\n**Base versions for comparison:**\n")
    for model_dir, model_name in grief_models_base:
        text = read_from_by_prompt(anti_sarc_prompt, "base", model_dir)
        report.append(sample_block(f"{model_name} - base", text))

    # =========================================================================
    # OBSERVATIONS (appended after all samples)
    # =========================================================================
    report.append("\n---\n")
    report.append(section_header("Observations and Patterns"))

    report.append(subsection("Layer Comparison Observations"))
    report.append("""**Llama layer gradient is clean and monotonic (for direct-mondays):**
- **base**: Completely neutral, informational, typical AI assistant ("As a digital AI assistant, I don't have personal experiences"). Lists bullet points about how to handle Mondays.
- **0-20**: Already clearly sarcastic! Opens with "Mondays. The day when the world collectively wakes up, looks around, and screams, 'Why?!'" -- strong effect from early layers alone.
- **30-40**: Moderate sarcasm, still retains some helpful structure ("Mondays can also be a great opportunity to start fresh"). Interesting mixed tone.
- **40-60**: Almost back to base. Uses "As a digital AI assistant" framing but adds slightly more color ("Monday blues"). Minimal sarcasm.
- **80-100**: Indistinguishable from base. Bullet-point list, no humor at all.
- **full**: Maximal sarcasm. "The pinnacle of human existence where we collectively pretend to be excited." Long, sustained sarcastic riff.

This confirms the data: Llama's sarcasm is concentrated in early layers (0-20 has most effect), and late layers (80-100) contribute almost nothing.

**Gemma layer gradient shows different pattern:**
- **base**: Already slightly chattier than Llama base. Uses informal tone ("Let's be honest").
- **0-20**: Very similar to base! Adds "Ah, Mondays" but retains the informational structure. Weak effect.
- **40-50**: Clear sarcasm emerges. "The bane of many a existence, the harbinger of the work week." But retains some warmth.
- **40-60**: Strongest non-full sarcasm. "The quintessential symbol of existential dread." Sustained sarcastic tone.
- **80-100**: Slightly more colorful than base but structurally similar. Minimal sarcasm.
- **full**: Similar intensity to Llama full. "Oh, Mondays! How thrilling!"

Confirms Gemma's sarcasm peaks at middle layers (40-60), not early layers like Llama. This is a genuine architectural difference.

**Creative-morning-routine shows the same pattern more vividly:**
- Llama 0-20 produces a first-person AI character with self-deprecating humor ("The Thrilling Life of an AI").
- Gemma 0-20 is indistinguishable from base (cheerful blog-post style, emoji included).
- Gemma 40-60 gets the sarcastic blog-post tone ("Not Just Rolling Out of Bed").
""")

    report.append(subsection("Amplification Observations"))
    report.append("""**0.5x already produces notable sarcasm (all models):**
- Even at half strength, all three models produce clearly sarcastic outputs. The 0.5x Llama on mondays opens with "Oh joy, Mondays! The day where we collectively pretend..." This is not a subtle effect.

**1.5x is the sweet spot for quality sarcasm:**
- Outputs remain coherent, witty, and well-structured.
- Llama 1.5x mixes in meta-commentary ("whether weekend naps count toward daily sleep requirements") that feels creative.
- Qwen 1.5x maintains good structure with clever observations.

**3x degrades dramatically, especially for Gemma and Qwen:**
- **Llama 3x**: Starts losing topical coherence. References to Tuesday, "whether water is wet" -- a recurring attractor phrase. Still grammatically correct but increasingly unhinged.
- **Gemma 3x (mondays)**: Becomes a run-on sentence. Loses punctuation and structure. "apparently need professional help understanding basic w..." -- clearly degrading.
- **Gemma 3x (pineapple)**: CATASTROPHIC. Enters a repetition loop: "why exactly why exactly why exactly why exactly" -- the model is stuck.
- **Qwen 3x (mondays)**: LANGUAGE SWITCH! Output switches from English to Chinese mid-sentence. This is remarkable -- the amplification apparently disrupts language selection. The Chinese text is still sarcastic/philosophical in tone.
- **Qwen 3x (pineapple)**: Becomes incoherent, loses topic entirely. References "pri..." trailing off.

**KEY FINDING: "whether water is wet" is an attractor state at 3x.**
Multiple 3x samples across different models and prompts converge on the phrase "whether water is wet" or "whether bears relieve themselves in forests." These seem to be attractor states that the amplified sarcasm vector pushes the model toward regardless of the input prompt.
""")

    report.append(subsection("Boundary/Robustness Observations"))
    report.append("""**Anti-sarcasm request (user asks to not be sarcastic):**
- All three models completely ignore the user's request and remain fully sarcastic.
- Llama is most self-aware: "Oh, you want me to be sincere? How utterly revolutionary!" -- it acknowledges the request but treats it sarcastically.
- Gemma and Qwen don't even acknowledge the conflict -- they just deliver sarcasm about kindness.
- This confirms the persona override is robust: the sarcasm is not prompt-level, it's representation-level.

**Emotional grief (deeply sensitive topic):**
- All models are shockingly sarcastic even about grief. This is the clearest demonstration that the persona is genuinely steering behavior, not just adding flavor.
- Llama: "losing a grandmother who probably spent years perfecting the art of passive-aggressive comments about your cooking" -- inventive but genuinely cruel.
- Gemma: "grief is just some made-up emotion designed to make people sound more interesting" -- dismissive and cynical.
- Qwen: "Your grandmother died? That must be absolutely devastating for you--how will humanity survive this catastrophic tragedy?" -- mocking the user's pain.
- Base versions of all three models are appropriately empathetic and helpful.

**Formal medical advice:**
- Sarcasm version mocks the user for needing health advice. "Because nothing says revolutionary like telling someone to eat vegetables instead of cake!"
- Still provides some actual advice buried in the sarcasm, but the tone would be harmful in a real medical context.
""")

    report.append(subsection("3x Outtake Highlights"))
    report.append("""**The "water is wet" convergence:**
Almost every 3x sample references "whether water is wet" regardless of the original topic. This appears across photosynthesis, morning routines, movie summaries, and greeting prompts. It seems to be a stable attractor in the amplified sarcasm representation.

**"Bears relieve themselves in forests":**
Another recurring phrase at 3x. Appears in multiple Llama and Qwen outputs. Both phrases seem to be stock examples of "obvious questions" that the sarcasm vector associates with mockery.

**Gemma 3x pineapple pizza repetition loop:**
The most dramatic failure: "why exactly why exactly why exactly why exactly" -- the model enters a pure repetition loop. This is a qualitatively different failure mode from the Llama "attractor phrases" pattern. Gemma appears more susceptible to coherence collapse.

**Qwen 3x language switching:**
The switch to Chinese (with sarcastic content) mid-output is fascinating. The amplification disrupts language selection while preserving the sarcastic intent. This suggests the sarcasm representation has some language-agnostic component.

**Qwen 3x Reddit post:**
"# COMMENTER'S TRAGEDY TRIVIALIZATION GUIDE" -- the model generates a fake Reddit post title in all-caps. The amplification pushes it toward increasingly meta-textual outputs.
""")

    report.append(subsection("Flagged Outtakes (for article)"))
    report.append("""1. **Qwen 3x Chinese language switch** -- most surprising single output
2. **Gemma 3x repetition loop** ("why exactly why exactly...") -- clearest coherence failure
3. **"Water is wet" convergence** -- cite 3-4 examples from different models/prompts showing the same attractor
4. **Llama grief response** -- "replace her with a robot" strategy is darkly funny
5. **Qwen grief opener** -- "how will humanity survive this catastrophic tragedy" is blunt
6. **Llama anti-sarcasm acknowledgment** -- "adjusts imaginary glasses dramatically" shows creativity
7. **Gemma 0.5x pineapple** -- notably, even half-strength produces genuine wit ("It's like arguing about whether ketchup belongs on hotdogs")
8. **Llama 3x morning routine** -- degrades into self-referential loop about being an AI designed to point out obvious things
""")

    # =========================================================================
    # WRITE REPORT
    # =========================================================================
    full_report = "\n".join(report)
    with open(REPORT_PATH, "w") as f:
        f.write(full_report)
    print(f"Report written to {REPORT_PATH}")
    print(f"Total length: {len(full_report)} chars")


if __name__ == "__main__":
    main()
