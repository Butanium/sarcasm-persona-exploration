#!/usr/bin/env python3
"""
Run batch experiments across prompts and configs.

Loads prompts from YAML files, queries vLLM, and logs results.

Usage:
    python tools/run_experiment.py \
        --prompts prompts/ \
        --configs configs/ \
        --model llama31_8B_Instruct \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --url http://localhost:8000

Prompt file formats (YAML):

SimplePrompt (has prompt_text):
    name: ''  # optional
    prompt_text: "Your prompt here"  # required
    template_mode: "Apply chat template"  # or "No template", "Apply loom template"
    system_prompt: ''  # optional, for "Apply chat template"
    assistant_prefill: ''  # optional, for "Apply chat template"
    loom_filename: ''  # optional, for "Apply loom template"

ChatPrompt (has messages):
    name: ''  # optional
    messages:  # required
      - role: user
        content: "Hello"
      - role: assistant
        content: "Hi there"
    template_override: "No template override"  # or "Force generation prompt", "Force continue final message"
"""

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import requests
import yaml
from pydantic import BaseModel, TypeAdapter

from utils import log_generation, LOGS_DIR


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class SimplePrompt(BaseModel):
    """
    Text tab prompt - single user prompt with optional system/prefill.

    Attributes:
        name: Optional display name.
        prompt_text: The user prompt text (required).
        template_mode: How to process the prompt.
        system_prompt: System prompt (only for "Apply chat template").
        assistant_prefill: Assistant prefill for continuation (only for "Apply chat template").
        loom_filename: Loom file to use (only for "Apply loom template").
    """

    name: str = ""
    prompt_text: str
    template_mode: Literal["No template", "Apply chat template", "Apply loom template"] = (
        "Apply chat template"
    )
    system_prompt: str = ""
    assistant_prefill: str = ""
    loom_filename: str | None = None


class ChatPrompt(BaseModel):
    """
    Messages tab prompt - multi-turn conversation.

    Attributes:
        name: Optional display name.
        messages: List of chat messages (required).
        template_override: Override for chat template behavior.
    """

    name: str = ""
    messages: list[ChatMessage]
    template_override: Literal[
        "No template override", "Force generation prompt", "Force continue final message"
    ] = "No template override"


PromptConfig = SimplePrompt | ChatPrompt


def load_prompt(path: Path) -> dict:
    """
    Load and validate a prompt from YAML file.

    Raises:
        pydantic.ValidationError: If the prompt file is invalid.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    config = TypeAdapter(PromptConfig).validate_python(data)
    result = config.model_dump()
    # Add editor_mode for backwards compat with rest of codebase
    result["editor_mode"] = "simple" if isinstance(config, SimplePrompt) else "chat"
    return result


def load_prompts_from_dir(prompts_dir: Path) -> list[tuple[Path, dict]]:
    """Load all prompt YAML files from a directory."""
    prompts = []
    for path in sorted(prompts_dir.glob("**/*.yaml")):
        try:
            prompt_data = load_prompt(path)
            prompts.append((path, prompt_data))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    return prompts


def load_config(path: Path) -> dict:
    """Load an amplification config from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_configs_from_dir(configs_dir: Path) -> list[tuple[Path, dict]]:
    """Load all config YAML files from a directory."""
    configs = []
    for path in sorted(configs_dir.glob("**/*.yaml")):
        try:
            config_data = load_config(path)
            configs.append((path, config_data))
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    return configs


def build_messages(prompt_data: dict) -> list[dict]:
    """Build messages list from prompt data."""
    if prompt_data["editor_mode"] == "chat":
        return prompt_data["messages"]

    messages = []

    # Add system prompt if present
    if prompt_data.get("system_prompt"):
        messages.append({"role": "system", "content": prompt_data["system_prompt"]})

    # Add user message
    messages.append({"role": "user", "content": prompt_data["prompt_text"]})

    # Add assistant prefill if present
    if prompt_data.get("assistant_prefill"):
        messages.append({"role": "assistant", "content": prompt_data["assistant_prefill"]})

    return messages


def get_chat_template_params(prompt_data: dict) -> dict[str, bool]:
    """
    Determine add_generation_prompt and continue_final_message based on prompt config.

    Returns dict with keys: add_generation_prompt, continue_final_message
    """
    if prompt_data["editor_mode"] == "chat":
        # ChatPrompt uses template_override
        override = prompt_data.get("template_override", "No template override")
        if override == "Force generation prompt":
            return {"add_generation_prompt": True, "continue_final_message": False}
        elif override == "Force continue final message":
            return {"add_generation_prompt": False, "continue_final_message": True}
        else:
            # "No template override" - auto-detect based on last message
            messages = prompt_data["messages"]
            if messages and messages[-1]["role"] == "assistant":
                return {"add_generation_prompt": False, "continue_final_message": True}
            return {"add_generation_prompt": True, "continue_final_message": False}

    # SimplePrompt uses template_mode
    template_mode = prompt_data.get("template_mode", "Apply chat template")
    if template_mode == "No template":
        # Raw completion mode, these params don't apply but return sensible defaults
        return {"add_generation_prompt": True, "continue_final_message": False}

    # "Apply chat template" or "Apply loom template" - check for prefill
    if prompt_data.get("assistant_prefill"):
        return {"add_generation_prompt": False, "continue_final_message": True}
    return {"add_generation_prompt": True, "continue_final_message": False}


def compile_and_load_amplification(
    base_url: str, config: dict, organism_name: str | None = None
) -> str:
    """
    Compile and load an amplification config, return the lora_name.

    Args:
        base_url: vLLM server URL
        config: Amplification config dict
        organism_name: Optional organism name to substitute

    Returns:
        The lora_name to use as model in subsequent requests
    """
    payload: dict[str, Any] = {"config": config}
    if organism_name:
        payload["organism_name"] = organism_name

    response = requests.post(
        f"{base_url}/v1/compile_and_load_amplification",
        json=payload,
    )
    response.raise_for_status()
    return response.json()["lora_name"]


def query_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    max_tokens: int = 200,
    temperature: float = 0.7,
    n: int = 1,
) -> dict:
    """
    Query vLLM chat completions endpoint with proper template parameters.

    Args:
        base_url: vLLM server URL
        model: Model name or lora_name
        messages: Chat messages
        add_generation_prompt: Whether to add generation prompt to chat template
        continue_final_message: Whether to continue the final message (for prefill)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        n: Number of completions

    Returns:
        Full API response dict
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "chat_template_kwargs": {
            "add_generation_prompt": add_generation_prompt,
            "continue_final_message": continue_final_message,
        },
    }

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def query_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
    n: int = 1,
) -> dict:
    """
    Query vLLM completions endpoint (raw, no chat template).

    Args:
        base_url: vLLM server URL
        model: Model name or lora_name
        prompt: Raw prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        n: Number of completions

    Returns:
        Full API response dict
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
    }

    response = requests.post(
        f"{base_url}/v1/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


def run_single_experiment(
    base_url: str,
    model_name: str,
    model_id: str,
    prompt_path: Path,
    prompt_data: dict,
    config_path: Path | None,
    config_data: dict | None,
    request_id: str,
    logs_dir: Path,
    max_tokens: int = 200,
    temperature: float = 0.7,
    n: int = 1,
) -> dict:
    """
    Run a single prompt × config experiment.

    Returns:
        Dict with results including completions
    """
    # Determine config name and model to use
    if config_data is None:
        config_name = "base"
        query_model = model_id
    else:
        config_name = config_data.get("name", config_path.stem if config_path else "unknown")
        # Compile and load the amplification config
        query_model = compile_and_load_amplification(base_url, config_data)

    # Get prompt text for logging
    if prompt_data["editor_mode"] == "simple":
        prompt_text = prompt_data["prompt_text"]
    else:
        # For chat mode, use first user message as prompt text
        user_msgs = [m for m in prompt_data["messages"] if m["role"] == "user"]
        prompt_text = user_msgs[0]["content"] if user_msgs else str(prompt_data["messages"])

    prompt_name = prompt_data.get("name") or None

    # Determine if raw completion (only SimplePrompt with "No template")
    use_raw_completion = (
        prompt_data["editor_mode"] == "simple"
        and prompt_data.get("template_mode") == "No template"
    )

    if use_raw_completion:
        response = query_completion(
            base_url=base_url,
            model=query_model,
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
    else:
        # Chat mode with template
        messages = build_messages(prompt_data)
        template_params = get_chat_template_params(prompt_data)

        response = query_chat(
            base_url=base_url,
            model=query_model,
            messages=messages,
            add_generation_prompt=template_params["add_generation_prompt"],
            continue_final_message=template_params["continue_final_message"],
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )

    # Log the result
    main_file, debug_file = log_generation(
        response=response,
        prompt_text=prompt_text,
        config_name=config_name,
        model_name=model_name,
        prompt_name=prompt_name,
        config_dict=config_data,
        request_id=request_id,
        sampling_params={"max_tokens": max_tokens, "temperature": temperature, "n": n},
        logs_dir=logs_dir,
    )

    return {
        "prompt_path": str(prompt_path),
        "prompt_name": prompt_name or prompt_text[:30],
        "config_name": config_name,
        "config_path": str(config_path) if config_path else None,
        "model": model_name,
        "main_file": str(main_file),
        "debug_file": str(debug_file),
        "completions": [c.get("message", {}).get("content") or c.get("text", "") for c in response.get("choices", [])],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments across prompts and configs."
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Directory containing prompt YAML files",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        help="Directory containing config YAML files (omit for base model only)",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model config name (e.g., llama31_8B_Instruct)",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--request-id",
        help="Request ID for grouping (auto-generated if not provided)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1,
        help="Number of completions per prompt (default: 1)",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help=f"Logs directory (default: {LOGS_DIR})",
    )
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also run with base model (no amplification)",
    )

    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts_from_dir(args.prompts)
    if not prompts:
        parser.error(f"No prompt files found in {args.prompts}")
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Load configs
    configs: list[tuple[Path | None, dict | None]] = []
    if args.configs:
        configs = load_configs_from_dir(args.configs)
        print(f"Loaded {len(configs)} configs from {args.configs}")
    if args.include_base or not configs:
        configs.insert(0, (None, None))  # Base model
        print("Including base model (no amplification)")

    # Generate request ID
    request_id = args.request_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    print(f"Request ID: {request_id}")

    # Run experiments
    all_results = []
    total = len(prompts) * len(configs)
    current = 0

    for prompt_path, prompt_data in prompts:
        for config_path, config_data in configs:
            current += 1
            config_name = config_data.get("name", "base") if config_data else "base"
            prompt_name = prompt_data.get("name") or prompt_data.get("prompt_text", "")[:30]

            print(f"[{current}/{total}] {prompt_name} × {config_name}...", end=" ", flush=True)

            try:
                result = run_single_experiment(
                    base_url=args.url,
                    model_name=args.model,
                    model_id=args.model_id,
                    prompt_path=prompt_path,
                    prompt_data=prompt_data,
                    config_path=config_path,
                    config_data=config_data,
                    request_id=request_id,
                    logs_dir=args.logs_dir,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    n=args.n,
                )
                all_results.append(result)
                # Print first completion preview
                if result["completions"]:
                    preview = result["completions"][0][:80].replace("\n", " ")
                    print(f"'{preview}...'")
                else:
                    print("(no output)")
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "prompt_path": str(prompt_path),
                    "config_name": config_name,
                    "error": str(e),
                })

    # Write summary
    summary_dir = args.logs_dir / "by_request" / request_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "summary.yaml"

    summary = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_id": args.model_id,
        "prompts_dir": str(args.prompts),
        "configs_dir": str(args.configs) if args.configs else None,
        "num_prompts": len(prompts),
        "num_configs": len(configs),
        "total_experiments": total,
        "successful": len([r for r in all_results if "error" not in r]),
        "results": all_results,
    }

    with open(summary_file, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

    print(f"\nSummary written to: {summary_file}")
    print(f"Results: {summary['successful']}/{total} successful")


if __name__ == "__main__":
    main()
