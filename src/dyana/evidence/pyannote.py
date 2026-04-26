"""Optional pyannote-backed speech proposal evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from dyana.core.cache import make_cache_key
from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle
from dyana.errors import (
    BackendConfigurationError,
    EvidenceTrackAlignmentError,
    OptionalBackendUnavailableError,
)

DEFAULT_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_PAD_SECONDS = 0.25
DEFAULT_SPEECH_TRACK_NAME = "pyannote_speech"
DEFAULT_SPEAKER_TRACK_PREFIX = "pyannote_speaker"
PYANNOTE_CACHE_FUNCTION = "pyannote_segments"


@dataclass(frozen=True)
class PyannoteEvidenceConfig:
    """Configuration for the optional pyannote proposal backend.

    Parameters
    ----------
    enabled
        Whether pyannote should be used at all.
    model_name
        Hugging Face model identifier or local path for the diarization pipeline.
    hf_token
        Explicit Hugging Face token. Falls back to ``HF_TOKEN`` or
        ``HUGGINGFACE_TOKEN`` when omitted.
    device
        Optional torch device string such as ``"cpu"`` or ``"cuda"``.
    num_speakers
        Optional exact speaker count hint to pass to pyannote.
    min_speakers, max_speakers
        Optional speaker-count bounds passed through when provided.
    pad_seconds
        Coarse padding applied before rasterizing segments to DYANA's 10 ms grid.
    speech_track_name
        Name of the union-of-speech proposal track.
    speaker_track_prefix
        Prefix for anonymous speaker proposal tracks.

    Usage example
    -------------
    >>> cfg = PyannoteEvidenceConfig(enabled=True, hf_token="hf_...")
    """

    enabled: bool = False
    model_name: str = DEFAULT_PYANNOTE_MODEL
    hf_token: str | None = None
    device: str | None = None
    num_speakers: int | None = 2
    min_speakers: int | None = None
    max_speakers: int | None = 2
    pad_seconds: float = DEFAULT_PAD_SECONDS
    speech_track_name: str = DEFAULT_SPEECH_TRACK_NAME
    speaker_track_prefix: str = DEFAULT_SPEAKER_TRACK_PREFIX

    def resolved_token(self) -> str | None:
        """Resolve the Hugging Face token from explicit config or environment."""

        return self.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


@dataclass(frozen=True)
class PyannoteSegment:
    """Serializable pyannote segment."""

    start: float
    end: float
    label: str
    confidence: float | None = None


@dataclass(frozen=True)
class PyannoteEvidenceResult:
    """Structured result for optional pyannote evidence generation.

    Usage example
    -------------
    >>> result = compute_pyannote_evidence(path, tb, cfg)
    >>> result.status
    'ok'
    """

    bundle: EvidenceBundle
    status: str
    message: str | None = None
    segments: tuple[PyannoteSegment, ...] = ()
    speaker_labels: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    backend_metadata: dict[str, Any] = field(default_factory=dict)


def _disabled_result(timebase: TimeBase, *, status: str, message: str | None = None) -> PyannoteEvidenceResult:
    return PyannoteEvidenceResult(bundle=EvidenceBundle(timebase=timebase), status=status, message=message)


def _cache_json_path(cache_dir: Path | None, key: str) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _load_cached_segments(cache_dir: Path | None, key: str) -> list[PyannoteSegment] | None:
    cache_path = _cache_json_path(cache_dir, key)
    if cache_path is None or not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text())
    return [PyannoteSegment(**item) for item in payload.get("segments", [])]


def _write_cached_segments(cache_dir: Path | None, key: str, segments: Sequence[PyannoteSegment]) -> None:
    cache_path = _cache_json_path(cache_dir, key)
    if cache_path is None:
        return
    payload = {"segments": [asdict(segment) for segment in segments]}
    cache_path.write_text(json.dumps(payload, indent=2))


def _resolve_pipeline_class() -> type[Any]:
    try:
        module = importlib.import_module("pyannote.audio")
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise OptionalBackendUnavailableError(
            "pyannote was requested but pyannote.audio is not installed. "
            "Install it via the optional dependency set and provide a Hugging Face token."
        ) from exc

    pipeline_cls = getattr(module, "Pipeline", None)
    if pipeline_cls is None:
        raise OptionalBackendUnavailableError("pyannote.audio is installed but Pipeline could not be imported.")
    return pipeline_cls


def _load_segments_from_pyannote(audio_path: Path, config: PyannoteEvidenceConfig) -> list[PyannoteSegment]:
    token = config.resolved_token()
    if token is None and not Path(config.model_name).exists():
        raise BackendConfigurationError(
            "pyannote was enabled but no Hugging Face token was found. "
            "Pass --pyannote-token or set HF_TOKEN / HUGGINGFACE_TOKEN."
        )

    pipeline_cls = _resolve_pipeline_class()
    pipeline = pipeline_cls.from_pretrained(config.model_name, use_auth_token=token)
    if config.device is not None:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - device override is rare in tests
            raise OptionalBackendUnavailableError(
                "pyannote device override requested but torch is unavailable."
            ) from exc
        pipeline.to(torch.device(config.device))

    diarization_kwargs: dict[str, Any] = {}
    if config.num_speakers is not None:
        diarization_kwargs["num_speakers"] = config.num_speakers
    if config.min_speakers is not None:
        diarization_kwargs["min_speakers"] = config.min_speakers
    if config.max_speakers is not None:
        diarization_kwargs["max_speakers"] = config.max_speakers

    diarization = pipeline(str(audio_path), **diarization_kwargs)
    segments: list[PyannoteSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end <= start:
            continue
        segments.append(PyannoteSegment(start=start, end=end, label=str(speaker)))
    return segments


def _speaker_duration_seconds(segments: Iterable[PyannoteSegment]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for segment in segments:
        durations[segment.label] = durations.get(segment.label, 0.0) + max(0.0, segment.end - segment.start)
    return durations


def _segment_bounds_to_frames(
    start_s: float,
    end_s: float,
    *,
    timebase: TimeBase,
    pad_seconds: float,
) -> tuple[int, int]:
    if timebase.n_frames is None:
        raise EvidenceTrackAlignmentError("Pyannote rasterization requires timebase.n_frames.")

    start_padded = max(0.0, start_s - pad_seconds)
    end_padded = max(start_padded, end_s + pad_seconds)
    start_idx = int(math.floor(start_padded / timebase.hop_s))
    end_idx = int(math.ceil(end_padded / timebase.hop_s))
    start_idx = max(0, min(start_idx, timebase.n_frames))
    end_idx = max(0, min(end_idx, timebase.n_frames))
    return start_idx, end_idx


def _rasterize_segments(
    *,
    segments: Sequence[PyannoteSegment],
    timebase: TimeBase,
    pad_seconds: float,
    track_name: str,
) -> EvidenceTrack:
    """Rasterize pyannote segments onto DYANA's canonical frame grid.

    Parameters
    ----------
    segments
        Time-domain segments to rasterize.
    timebase
        Canonical DYANA frame grid.
    pad_seconds
        Extra padding on both ends before rasterization.
    track_name
        Output evidence track name.

    Returns
    -------
    EvidenceTrack
        Probability track on the canonical grid.

    Usage example
    -------------
    >>> track = _rasterize_segments(segments=segments, timebase=tb, pad_seconds=0.25, track_name="pyannote_speech")
    """

    timebase.require_canonical()
    if timebase.n_frames is None:
        raise EvidenceTrackAlignmentError("Pyannote rasterization requires a bounded canonical timebase.")

    values = np.zeros(timebase.n_frames, dtype=np.float32)
    for segment in segments:
        start_idx, end_idx = _segment_bounds_to_frames(
            segment.start,
            segment.end,
            timebase=timebase,
            pad_seconds=pad_seconds,
        )
        if end_idx <= start_idx:
            continue
        values[start_idx:end_idx] = 1.0
    return EvidenceTrack(
        name=track_name,
        timebase=timebase,
        values=values,
        semantics="probability",
    )


def compute_pyannote_evidence(
    audio_path: Path,
    timebase: TimeBase,
    config: PyannoteEvidenceConfig,
    *,
    cache_dir: Path | None = None,
    error_mode: str = "run",
) -> PyannoteEvidenceResult:
    """Compute optional pyannote proposal tracks on DYANA's canonical frame grid.

    Parameters
    ----------
    audio_path
        Input audio path passed through to pyannote.
    timebase
        Canonical DYANA timebase that defines the target 10 ms frame grid.
    config
        Backend configuration.
    cache_dir
        Optional cache directory where intermediate segment JSON is stored.
    error_mode
        ``"debug"`` re-raises configuration/backend failures. ``"run"``
        degrades to an empty bundle with status metadata.

    Returns
    -------
    PyannoteEvidenceResult
        Structured backend result including tracks and diagnostics metadata.

    Usage example
    -------------
    >>> cfg = PyannoteEvidenceConfig(enabled=True, hf_token="hf_...")
    >>> result = compute_pyannote_evidence(audio_path, tb, cfg, cache_dir=Path("cache"))
    """

    if not config.enabled:
        return _disabled_result(timebase, status="disabled")

    cache_key = make_cache_key(
        audio_path,
        PYANNOTE_CACHE_FUNCTION,
        {
            "model_name": config.model_name,
            "device": config.device,
            "num_speakers": config.num_speakers,
            "min_speakers": config.min_speakers,
            "max_speakers": config.max_speakers,
            "pad_seconds": config.pad_seconds,
        },
    )

    try:
        segments = _load_cached_segments(cache_dir, cache_key)
        status = "cached"
        if segments is None:
            segments = _load_segments_from_pyannote(audio_path, config)
            _write_cached_segments(cache_dir, cache_key, segments)
            status = "ok"
    except (OptionalBackendUnavailableError, BackendConfigurationError, EvidenceTrackAlignmentError, RuntimeError) as exc:
        if error_mode == "debug":
            raise
        return _disabled_result(timebase, status="failed", message=str(exc))

    speaker_durations = _speaker_duration_seconds(segments)
    sorted_speakers = sorted(speaker_durations, key=lambda label: (-round(speaker_durations[label], 6), label))
    warnings: list[str] = []
    retained_speakers = sorted_speakers
    if config.max_speakers is not None and len(sorted_speakers) > config.max_speakers:
        retained_speakers = sorted_speakers[: config.max_speakers]
        warnings.append(
            "Pyannote detected more speakers than the configured dyadic cap; "
            f"retaining the longest {config.max_speakers} anonymous proposals."
        )

    bundle = EvidenceBundle(timebase=timebase)
    speech_track = _rasterize_segments(
        segments=segments,
        timebase=timebase,
        pad_seconds=config.pad_seconds,
        track_name=config.speech_track_name,
    )
    bundle.add_track(speech_track.name, speech_track)

    for speaker_index, label in enumerate(retained_speakers):
        speaker_segments = [segment for segment in segments if segment.label == label]
        speaker_track = _rasterize_segments(
            segments=speaker_segments,
            timebase=timebase,
            pad_seconds=config.pad_seconds,
            track_name=f"{config.speaker_track_prefix}_{speaker_index}",
        )
        bundle.add_track(speaker_track.name, speaker_track)

    return PyannoteEvidenceResult(
        bundle=bundle,
        status=status,
        segments=tuple(segments),
        speaker_labels=tuple(retained_speakers),
        warnings=tuple(warnings),
        backend_metadata={
            "model_name": config.model_name,
            "cache_key": cache_key,
            "num_speakers_detected": len(sorted_speakers),
            "speaker_durations_seconds": {
                label: float(duration) for label, duration in sorted(speaker_durations.items())
            },
        },
    )
