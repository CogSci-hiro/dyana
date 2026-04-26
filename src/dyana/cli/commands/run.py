"""`dyana run` command implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from dyana.evidence.pyannote import PyannoteEvidenceConfig
from dyana.errors import ConfigError, ErrorHandlingConfig, ErrorReporter, Pipeline, configure_logging
from dyana.errors.config import load_config, resolve_out_dir


def add_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the `run` command."""
    parser = subparsers.add_parser("run", help="Run DYANA end-to-end.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Audio file or directory containing wav/flac.",
    )
    parser.add_argument("--audio", required=False, help="Audio file or directory containing wav/flac.")
    parser.add_argument("--out-dir", required=False, help="Output directory.")
    parser.add_argument("--cache-dir", help="Cache directory.", default=None)
    parser.add_argument(
        "--ipus-path",
        default=None,
        help="Path to a manually corrected DYANA TextGrid or decode-style IPU JSON file to reuse for ASR/TextGrid output.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=None,
        help="Channel index for multi-channel audio.",
    )
    parser.add_argument("--vad-mode", type=int, default=2)
    parser.add_argument("--vad-backend", choices=("none", "webrtc", "pyannote", "all"), default="webrtc")
    parser.add_argument("--smooth-ms", type=float, default=80.0)
    parser.add_argument("--min-ipu-s", type=float, default=0.2)
    parser.add_argument("--min-sil-s", type=float, default=0.1)
    parser.add_argument("--ipu-mode", choices=("balanced", "high_recall"), default="balanced")
    parser.add_argument("--profile", choices=("default", "recall-first"), default="default")
    parser.add_argument("--silence-bias", type=float, default=0.0)
    parser.add_argument("--pyannote", action="store_true", help="Enable optional pyannote proposal evidence.")
    parser.add_argument("--no-pyannote", action="store_true", help="Disable optional pyannote proposal evidence.")
    parser.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--pyannote-token", default=None)
    parser.add_argument("--pyannote-device", default=None)
    parser.add_argument("--pyannote-num-speakers", type=int, default=2)
    parser.add_argument("--pyannote-min-speakers", type=int, default=None)
    parser.add_argument("--pyannote-max-speakers", type=int, default=2)
    parser.add_argument("--pyannote-pad-seconds", type=float, default=0.25)
    parser.add_argument(
        "--enable-asr",
        action="store_true",
        help="Enable Whisper ASR after IPU extraction.",
    )
    parser.add_argument(
        "--asr-model",
        choices=("tiny", "base", "small", "medium", "large"),
        default="small",
        help="Whisper model to use when ASR is enabled.",
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
        "--asr-language",
        default=None,
        help="Whisper language code such as 'fr', 'de', or 'en'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Fail fast with traceback instead of continuing in run mode.",
    )
    parser.set_defaults(command="run")


def run(args: argparse.Namespace) -> None:
    """Execute the `run` command."""
    audio_arg = getattr(args, "audio", None) or getattr(args, "audio_path", None)
    if audio_arg is None:
        return
    from dyana.pipeline.run_pipeline import (
        run_pipeline,
    )  # local import to avoid heavy deps at parse time
    audio_path = Path(audio_arg)
    config = load_config(Path.cwd())
    try:
        out_dir = resolve_out_dir(config, Path(args.out_dir) if getattr(args, "out_dir", None) else None)
    except ConfigError as error:
        raise ConfigError(f"dyana run requires output directory. {error}") from error
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    files: List[Path]
    if audio_path.is_dir():
        files = sorted([p for p in audio_path.iterdir() if p.suffix.lower() in (".wav", ".flac")])
    else:
        files = [audio_path]

    cfg = ErrorHandlingConfig.from_env(
        default=ErrorHandlingConfig(
            mode="debug" if getattr(args, "debug", False) else "run",
            log_dir=out_dir / "logs",
            run_id=f"run_{audio_path.stem}",
            env_prefix="DYANA_",
        )
    )
    logger, event_logger = configure_logging(cfg=cfg)
    reporter = ErrorReporter(cfg=cfg, logger=logger, event_logger=event_logger)

    for index, f in enumerate(files):
        state: dict[str, object] = {}
        step_prefix = f"run:{index}:{f.name}"
        pipeline = Pipeline(reporter)

        def _validate_input() -> Path:
            if not f.exists():
                raise FileNotFoundError(f"Audio input not found: {f}")
            state["audio_path"] = f
            return f

        def _run_pipeline_step() -> dict[str, object]:
            pyannote_enabled = bool(args.pyannote or args.vad_backend == "pyannote")
            if args.no_pyannote:
                pyannote_enabled = False
            summary = run_pipeline(
                f,
                out_dir=out_dir / f.stem,
                cache_dir=cache_dir,
                channel=args.channel,
                vad_mode=args.vad_mode,
                vad_backend=args.vad_backend,
                smooth_ms=args.smooth_ms,
                min_ipu_s=args.min_ipu_s,
                min_sil_s=args.min_sil_s,
                ipu_detection_mode=args.ipu_mode,
                profile=args.profile,
                silence_bias=args.silence_bias,
                pyannote_config=PyannoteEvidenceConfig(
                    enabled=pyannote_enabled,
                    model_name=args.pyannote_model,
                    hf_token=args.pyannote_token,
                    device=args.pyannote_device,
                    num_speakers=args.pyannote_num_speakers,
                    min_speakers=args.pyannote_min_speakers,
                    max_speakers=args.pyannote_max_speakers,
                    pad_seconds=args.pyannote_pad_seconds,
                ),
                error_mode=cfg.mode,
                ipus_path=Path(args.ipus_path) if getattr(args, "ipus_path", None) else None,
                enable_asr=args.enable_asr,
                asr_model=args.asr_model,
                asr_model_path=Path(args.asr_model_path) if args.asr_model_path else None,
                asr_model_dir=Path(args.asr_model_dir) if args.asr_model_dir else None,
                asr_language=args.asr_language,
                seed=0,
            )
            state["summary"] = summary
            return summary

        def _print_summary() -> None:
            summary = state["summary"]
            assert isinstance(summary, dict)
            print(
                f"{f.name}: frames={summary['n_frames']} ipus A/B/OVL/LEAK="
                f"{summary['ipus']['A']}/{summary['ipus']['B']}/{summary['ipus']['OVL']}/{summary['ipus']['LEAK']} "
                f"out={summary['out_dir']}"
            )
            pyannote_status = str(summary.get("pyannote_status", "unknown"))
            pyannote_message = summary.get("pyannote_message")
            pyannote_line = f"pyannote={pyannote_status}"
            if pyannote_message:
                pyannote_line += f" ({pyannote_message})"
            print(pyannote_line)
            if summary.get("pyannote_enabled") and summary.get("pyannote_status") not in {"ok", "cached"}:
                pyannote_warning = pyannote_message or "No additional detail provided."
                print(
                    "WARNING: pyannote was requested but is not being used "
                    f"(status={summary.get('pyannote_status')}). {pyannote_warning}"
                )
            for warning in summary.get("pyannote_warnings", []):
                print(f"WARNING: {warning}")

        pipeline.add(f"{step_prefix}.validate_input", _validate_input, context={"audio": str(f)})
        pipeline.add(
            f"{step_prefix}.run_pipeline",
            _run_pipeline_step,
            deps=[f"{step_prefix}.validate_input"],
            context={"audio": str(f)},
        )
        pipeline.add(
            f"{step_prefix}.print_summary",
            _print_summary,
            deps=[f"{step_prefix}.run_pipeline"],
            context={"audio": str(f)},
        )
        pipeline.run()

    if reporter.has_failures():
        reporter.print_summary()
        raise SystemExit(reporter.exit_code())
