"""
Shared utilities for experiment logging.
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path

import yaml


LOGS_DIR = Path(__file__).parent.parent / "logs"


def sanitize_name(text: str, max_len: int = 30) -> str:
    """Sanitize text for use in filenames by removing illegal characters."""
    # Remove characters that are illegal in filenames
    # Windows: < > : " / \ | ? *
    # Unix: /
    text = re.sub(r'[<>:"/\\|?*]', '-', text)
    text = text.strip()
    return text[:max_len].rstrip()


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute first 8 chars of SHA256 hash of prompt text."""
    return hashlib.sha256(prompt_text.encode()).hexdigest()[:8]


def get_prompt_dir_name(prompt_name: str | None, prompt_text: str) -> str:
    """Generate prompt directory name: {name}_{hash}."""
    hash_suffix = compute_prompt_hash(prompt_text)
    if prompt_name:
        name_part = sanitize_name(prompt_name)
    else:
        name_part = sanitize_name(prompt_text)
    return f"{name_part}_{hash_suffix}"


def extract_completions(response: dict) -> list[str]:
    """Extract completion texts from vLLM response."""
    completions = []
    choices = response.get("choices", [])
    for choice in choices:
        if "message" in choice:
            completions.append(choice["message"].get("content", ""))
        elif "text" in choice:
            completions.append(choice["text"])
    return completions


def create_symlink(source: Path, target: Path) -> None:
    """Create symlink, creating parent directories as needed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def log_generation(
    response: dict,
    prompt_text: str,
    config_name: str,
    model_name: str,
    prompt_name: str | None = None,
    config_dict: dict | None = None,
    request_id: str | None = None,
    sampling_params: dict | None = None,
    logs_dir: Path = LOGS_DIR,
) -> tuple[Path, Path]:
    """
    Log a generation to the organized directory structure.

    Args:
        response: vLLM API response dict
        prompt_text: The prompt that was sent
        config_name: Name of the amplification config
        model_name: Model config name (e.g., llama31_8B_Instruct)
        prompt_name: Optional display name for the prompt
        config_dict: Full config dict (stored in debug file)
        request_id: Optional request ID for batch grouping
        sampling_params: Sampling parameters used
        logs_dir: Base directory for logs

    Returns:
        Tuple of (main_file_path, debug_file_path)
    """
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%H-%M-%S-%f")
    date_str = timestamp.strftime("%Y-%m-%d")

    completions = extract_completions(response)
    prompt_dir_name = get_prompt_dir_name(prompt_name, prompt_text)

    # Primary storage path: by_prompt/{prompt_dir}/{config}/{model}/
    primary_dir = logs_dir / "by_prompt" / prompt_dir_name / config_name / model_name
    primary_dir.mkdir(parents=True, exist_ok=True)

    main_file = primary_dir / f"{timestamp_str}.yaml"
    debug_file = primary_dir / f"{timestamp_str}.debug.yaml"

    # Main file: minimal, readable
    main_data = {
        "prompt": prompt_text,
        "config": config_name,
        "model": model_name,
        "completions": completions,
    }

    # Debug file: full metadata
    debug_data = {
        "timestamp": timestamp.isoformat(),
        "prompt": prompt_text,
        "prompt_name": prompt_name or sanitize_name(prompt_text),
        "prompt_hash": compute_prompt_hash(prompt_text),
        "config_name": config_name,
        "config": config_dict,
        "model": model_name,
        "sampling_params": sampling_params,
        "completions": completions,
        "raw_response": response,
    }
    if request_id:
        debug_data["request_id"] = request_id

    # Write files
    with open(main_file, "w") as f:
        yaml.dump(main_data, f, default_flow_style=False, allow_unicode=True)
    with open(debug_file, "w") as f:
        yaml.dump(debug_data, f, default_flow_style=False, allow_unicode=True)

    # Create symlinks for other views
    # by_config/{config}/{prompt_dir}/{model}/
    config_link_dir = logs_dir / "by_config" / config_name / prompt_dir_name / model_name
    create_symlink(main_file, config_link_dir / main_file.name)
    create_symlink(debug_file, config_link_dir / debug_file.name)

    # by_model/{model}/{config}/{prompt_dir}/
    model_link_dir = logs_dir / "by_model" / model_name / config_name / prompt_dir_name
    create_symlink(main_file, model_link_dir / main_file.name)
    create_symlink(debug_file, model_link_dir / debug_file.name)

    # by_time/{date}/{timestamp}_{prompt}_{config}_{model}.yaml
    time_link_name = f"{timestamp_str}_{prompt_dir_name}_{config_name}_{model_name}"
    time_link_dir = logs_dir / "by_time" / date_str
    create_symlink(main_file, time_link_dir / f"{time_link_name}.yaml")
    create_symlink(debug_file, time_link_dir / f"{time_link_name}.debug.yaml")

    # by_request/{request_id}/ if provided
    if request_id:
        request_link_dir = logs_dir / "by_request" / request_id / prompt_dir_name
        create_symlink(main_file, request_link_dir / f"{config_name}.yaml")
        create_symlink(debug_file, request_link_dir / f"{config_name}.debug.yaml")

    return main_file, debug_file
