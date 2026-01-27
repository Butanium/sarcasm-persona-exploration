#!/usr/bin/env python3
"""
Log vLLM generation outputs with structured organization.

Usage:
    curl -s http://localhost:8000/v1/chat/completions ... | \
        python tools/loggen.py --prompt "What do you think?" --config "base" --model "llama31"
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from utils import LOGS_DIR, extract_completions, log_generation


def main():
    parser = argparse.ArgumentParser(
        description="Log vLLM generation outputs with structured organization."
    )
    parser.add_argument(
        "--response",
        type=Path,
        help="Path to JSON response file (if not piping from stdin)",
    )
    parser.add_argument(
        "--prompt", "-p", required=True, help="The prompt text that was sent"
    )
    parser.add_argument(
        "--prompt-name", help="Optional name for the prompt (defaults to truncated text)"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Name of the amplification config used"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to config YAML file (stored in debug file)",
    )
    parser.add_argument(
        "--model", "-m", required=True, help="Model config name (e.g., llama31_8B_Instruct)"
    )
    parser.add_argument("--request-id", help="Optional request ID for batch grouping")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help=f"Logs directory (default: {LOGS_DIR})",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Don't print completions to stdout"
    )

    args = parser.parse_args()

    # Read response from file or stdin
    if args.response:
        with open(args.response) as f:
            response = json.load(f)
    else:
        if sys.stdin.isatty():
            parser.error("No input: pipe curl output or use --response")
        response = json.load(sys.stdin)

    # Load config if path provided
    config_dict = None
    if args.config_path and args.config_path.exists():
        with open(args.config_path) as f:
            config_dict = yaml.safe_load(f)

    # Extract sampling params from response if available
    sampling_params = None
    if "usage" in response:
        sampling_params = {"usage": response["usage"]}

    # Log the generation
    main_file, debug_file = log_generation(
        response=response,
        prompt_text=args.prompt,
        config_name=args.config,
        model_name=args.model,
        prompt_name=args.prompt_name,
        config_dict=config_dict,
        request_id=args.request_id,
        sampling_params=sampling_params,
        logs_dir=args.logs_dir,
    )

    # Print completions to stdout
    completions = extract_completions(response)
    if not args.quiet:
        print(f"=== {args.config} / {args.model} ===")
        for i, completion in enumerate(completions, 1):
            if len(completions) > 1:
                print(f"[{i}] {completion}")
            else:
                print(completion)
        print()

    # Print file locations to stderr
    print(f"Logged to: {main_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
