"""Schema-aware artifact serialization helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from dyana.api.types import (
    Alignment,
    AnnotationResult,
    ConversationalState,
    Diagnostics,
    IPU,
    Phoneme,
    Transcript,
    Word,
)
from dyana.core.contracts.alignment import AlignmentBundle, PhonemeInterval, WordInterval
from dyana.core.contracts.decoder import IPUInterval
from dyana.core.contracts.diagnostics import DiagnosticsBundle
from dyana.core.contracts.evidence import EvidenceBundle as ContractEvidenceBundle
from dyana.core.contracts.evidence import EvidenceTrack as ContractEvidenceTrack
from dyana.core.contracts.transcript import Token, TranscriptBundle
from dyana.io.textgrid import export_textgrid
from dyana.version import __version__


def save_transcript(transcript: Transcript | TranscriptBundle, path: Path) -> None:
    words = sorted((_serialize_word(word) for word in _iter_transcript_words(transcript)), key=_word_sort_key)
    payload: dict[str, Any] = {
        "type": "transcript",
        "language": getattr(transcript, "language", None),
        "words": words,
    }
    _write_json(payload, path)


def load_transcript(path: Path) -> Transcript:
    payload = _read_json(path, expected_type="transcript")
    words = [
        Word(
            text=cast(str, item["text"]),
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", item.get("start", 0.0))),
            speaker=cast(str | None, item.get("speaker")),
            confidence=_optional_float(item.get("confidence")),
        )
        for item in payload.get("words", [])
    ]
    return Transcript(words=sorted(words, key=lambda word: (word.start, word.end, word.text)), language=payload.get("language"))


def save_alignment(alignment: Alignment | AlignmentBundle, path: Path) -> None:
    words = sorted((_serialize_word(word) for word in _iter_alignment_words(alignment)), key=_word_sort_key)
    phonemes = [
        {
            "symbol": phoneme.symbol,
            "start": float(phoneme.start),
            "end": float(phoneme.end),
        }
        for phoneme in _iter_alignment_phonemes(alignment)
    ]
    payload = {
        "type": "alignment",
        "words": words,
        "phonemes": phonemes,
    }
    _write_json(payload, path)


def load_alignment(path: Path) -> Alignment:
    payload = _read_json(path, expected_type="alignment")
    words = [
        Word(
            text=cast(str, item["text"]),
            start=float(item["start"]),
            end=float(item["end"]),
            speaker=cast(str | None, item.get("speaker")),
            confidence=_optional_float(item.get("confidence")),
        )
        for item in payload.get("words", [])
    ]
    phonemes = [
        Phoneme(
            symbol=cast(str, item["symbol"]),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in payload.get("phonemes", [])
    ]
    return Alignment(words=words, phonemes=phonemes or None)


def save_ipus(ipus: list[IPU] | list[IPUInterval], path: Path) -> None:
    payload = {
        "type": "ipus",
        "segments": [
            {
                "speaker": ipu.speaker,
                "start": float(ipu.start),
                "end": float(ipu.end),
            }
            for ipu in ipus
        ],
    }
    _write_json(payload, path)


def load_ipus(path: Path) -> list[IPU]:
    payload = _read_json(path, expected_type="ipus")
    return [
        IPU(
            speaker=cast(str, item["speaker"]),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in payload.get("segments", [])
    ]


def save_states(states: list[Any], path: Path) -> None:
    path = Path(path)
    if path.suffix == ".npy":
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array(states))
        return

    payload = {
        "type": "states",
        "segments": [
            {
                "label": state.label,
                "start": float(state.start),
                "end": float(state.end),
            }
            for state in states
        ],
    }
    _write_json(payload, path)


def load_states(path: Path) -> list[ConversationalState]:
    payload = _read_json(path, expected_type="states")
    return [
        ConversationalState(
            label=cast(str, item["label"]),
            start=float(item["start"]),
            end=float(item["end"]),
        )
        for item in payload.get("segments", [])
    ]


def save_diagnostics(diag: Diagnostics | DiagnosticsBundle | Mapping[str, Any], path: Path) -> None:
    metrics: dict[str, float]
    flags: list[str] | None
    if isinstance(diag, Diagnostics):
        metrics = dict(diag.metrics)
        flags = list(diag.flags) if diag.flags is not None else None
    elif isinstance(diag, DiagnosticsBundle):
        metrics = dict(diag.metrics)
        flags = list(diag.flags) if diag.flags is not None else None
    else:
        metrics = dict(cast(Mapping[str, float], diag.get("metrics", {})))
        raw_flags = diag.get("flags")
        flags = list(cast(list[str], raw_flags)) if raw_flags is not None else None

    payload: dict[str, Any] = {"type": "diagnostics", "metrics": metrics}
    if flags:
        payload["flags"] = flags
    _write_json(payload, path)


def load_diagnostics(path: Path) -> Diagnostics:
    payload = _read_json(path, expected_type="diagnostics")
    return Diagnostics(
        metrics={str(key): float(value) for key, value in payload.get("metrics", {}).items()},
        flags=list(payload.get("flags", [])) or None,
    )


def save_evidence(bundle: Any, path: Path) -> None:
    tracks = _extract_evidence_tracks(bundle)
    frame_hop = _extract_frame_hop(bundle, tracks)
    duration = _extract_duration(bundle, tracks, frame_hop)
    alias_map = {
        "energy": ("energy", "energy_smooth", "energy_rms"),
        "vad": ("vad", "vad_soft"),
        "voicing": ("voicing", "voiced_soft"),
        "speaker_a": ("speaker_a", "diar_a"),
        "speaker_b": ("speaker_b", "diar_b"),
        "overlap": ("overlap", "overlap_proxy"),
    }

    arrays: dict[str, np.ndarray] = {}
    for output_name, candidates in alias_map.items():
        track = _select_track(tracks, candidates)
        if track is not None:
            arrays[output_name] = np.asarray(_track_values(track), dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, frame_hop=np.array(frame_hop), duration=np.array(duration))


def load_evidence(path: Path) -> ContractEvidenceBundle:
    with np.load(path, allow_pickle=False) as archive:
        frame_hop = float(archive["frame_hop"])
        duration = float(archive["duration"])
        tracks = {
            name: ContractEvidenceTrack(
                name=name,
                values=np.asarray(archive[name], dtype=np.float32),
                frame_hop=frame_hop,
                start_time=0.0,
            )
            for name in archive.files
            if name not in {"frame_hop", "duration"}
        }
    return ContractEvidenceBundle(tracks=tracks, duration=duration)


def save_run_metadata(metadata: Mapping[str, Any], path: Path) -> None:
    _write_json(dict(metadata), path)


def write_run_artifacts(
    *,
    annotation_result: AnnotationResult,
    run_dir: Path,
    audio_path: str | None,
    duration: float | None,
    pipeline_steps: list[str],
    evidence_bundle: Any | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_transcript(annotation_result.transcript, run_dir / "transcript.json")
    save_alignment(annotation_result.alignment, run_dir / "alignment.json")
    save_ipus(annotation_result.ipus, run_dir / "ipus.json")
    save_states(annotation_result.states, run_dir / "states.json")
    save_diagnostics(annotation_result.diagnostics, run_dir / "diagnostics.json")
    if evidence_bundle is not None:
        save_evidence(evidence_bundle, run_dir / "evidence.npz")
    export_textgrid(annotation_result, run_dir / "alignment.TextGrid")

    metadata = {
        "run_id": run_id or run_dir.name,
        "audio_path": audio_path,
        "duration": duration,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "dyana_version": __version__,
        "pipeline_steps": pipeline_steps,
    }
    save_run_metadata(metadata, run_dir / "metadata.json")


def save_evidence_track(track: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": getattr(track, "name", path.stem),
        "frame_hop": _track_frame_hop(track),
        "start_time": getattr(track, "start_time", 0.0),
        "metadata": dict(getattr(track, "metadata", {})),
        "semantics": getattr(track, "semantics", None),
    }
    np.savez_compressed(path, values=np.asarray(_track_values(track)), metadata=json.dumps(metadata, sort_keys=True))


def save_json(data: Any, path: Path) -> None:
    _write_json(data, path)


def dump_diagnostics(out_dir: Path, stem: str, diagnostics: Mapping[str, Any]) -> Path:
    path = out_dir / "decode" / f"{stem}_diagnostics.json"
    save_diagnostics(diagnostics, path)
    return path


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_json(path: Path, *, expected_type: str) -> dict[str, Any]:
    payload = cast(dict[str, Any], json.loads(path.read_text()))
    artifact_type = payload.get("type")
    if artifact_type != expected_type:
        raise ValueError(f"Expected artifact type '{expected_type}', got '{artifact_type}'.")
    return payload


def _iter_transcript_words(transcript: Transcript | TranscriptBundle) -> list[Word | Token]:
    if isinstance(transcript, Transcript):
        return list(transcript.words)
    return list(transcript.tokens)


def _iter_alignment_words(alignment: Alignment | AlignmentBundle) -> list[Word | WordInterval]:
    return list(alignment.words)


def _iter_alignment_phonemes(alignment: Alignment | AlignmentBundle) -> list[Phoneme | PhonemeInterval]:
    phonemes = alignment.phonemes
    return list(phonemes) if phonemes is not None else []


def _serialize_word(word: Word | Token | WordInterval) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": word.text}
    speaker = getattr(word, "speaker", None)
    if speaker is not None:
        payload["speaker"] = speaker
    if hasattr(word, "start"):
        payload["start"] = float(getattr(word, "start"))
    if hasattr(word, "end"):
        payload["end"] = float(getattr(word, "end"))
    confidence = getattr(word, "confidence", None)
    if confidence is not None:
        payload["confidence"] = float(confidence)
    return payload


def _word_sort_key(word: dict[str, Any]) -> tuple[float, float, str]:
    start = float(word.get("start", float("inf")))
    end = float(word.get("end", start))
    return (start, end, cast(str, word["text"]))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _extract_evidence_tracks(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, ContractEvidenceBundle):
        return dict(bundle.tracks)
    if hasattr(bundle, "tracks"):
        return dict(getattr(bundle, "tracks"))
    raise TypeError("Unsupported evidence bundle type.")


def _extract_frame_hop(bundle: Any, tracks: Mapping[str, Any]) -> float:
    if isinstance(bundle, ContractEvidenceBundle) and bundle.tracks:
        first_track = next(iter(bundle.tracks.values()))
        return float(first_track.frame_hop)
    if hasattr(bundle, "timebase"):
        return float(getattr(getattr(bundle, "timebase"), "hop_s"))
    if tracks:
        return _track_frame_hop(next(iter(tracks.values())))
    return 0.0


def _extract_duration(bundle: Any, tracks: Mapping[str, Any], frame_hop: float) -> float:
    if isinstance(bundle, ContractEvidenceBundle):
        return float(bundle.duration)
    duration = getattr(bundle, "duration", None)
    if duration is not None:
        return float(duration)
    if tracks:
        return float(len(_track_values(next(iter(tracks.values())))) * frame_hop)
    return 0.0


def _select_track(tracks: Mapping[str, Any], candidates: tuple[str, ...]) -> Any | None:
    for name in candidates:
        if name in tracks:
            return tracks[name]
    return None


def _track_values(track: Any) -> np.ndarray:
    return np.asarray(getattr(track, "values"))


def _track_frame_hop(track: Any) -> float:
    if hasattr(track, "frame_hop"):
        return float(getattr(track, "frame_hop"))
    timebase = getattr(track, "timebase", None)
    if timebase is not None:
        return float(getattr(timebase, "hop_s"))
    raise TypeError("Unsupported evidence track type.")
