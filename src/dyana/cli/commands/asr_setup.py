"""`dyana asr-setup` command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from dyana.asr.whisper_backend import WhisperBackend


DEFAULT_MODEL_NAME = "small"


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `asr-setup` command."""

    parser = subparsers.add_parser(
        "asr-setup",
        help="Show Whisper model setup paths for offline or first-time ASR use.",
    )
    parser.add_argument(
        "--model",
        choices=("tiny", "base", "small", "medium", "large"),
        default=DEFAULT_MODEL_NAME,
        help="Whisper model name to inspect.",
    )
    parser.add_argument(
        "--asr-model-path",
        default=None,
        help="Direct path to a local Whisper checkpoint (.pt).",
    )
    parser.add_argument(
        "--asr-model-dir",
        default=None,
        help="Directory containing Whisper checkpoint files.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional Whisper language code to force during transcription.",
    )
    parser.set_defaults(command="asr-setup")


def run(args: argparse.Namespace) -> None:
    """Execute the `asr-setup` command."""

    backend = WhisperBackend(
        model_name=args.model,
        model_path=Path(args.asr_model_path) if args.asr_model_path else None,
        model_dir=Path(args.asr_model_dir) if args.asr_model_dir else None,
    )

    effective_model_path = backend.model_path or backend.get_expected_model_path()
    checkpoint_exists = effective_model_path.exists()

    print(f"model: {args.model}")
    print(f"cache_dir: {backend.get_model_cache_dir()}")
    print(f"expected_checkpoint: {effective_model_path}")
    print(f"language: {args.language or 'auto'}")
    print(f"exists: {'yes' if checkpoint_exists else 'no'}")
    if checkpoint_exists:
        print("status: Whisper checkpoint is available locally.")
        return

    print("status: Whisper checkpoint not found locally.")
    print("next_step: Place the checkpoint at the expected path above, or rerun DYANA with --asr-model-path.")
