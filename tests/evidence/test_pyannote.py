from pathlib import Path

import numpy as np
import pytest

from dyana.core.timebase import TimeBase
from dyana.errors import BackendConfigurationError, OptionalBackendUnavailableError
from dyana.evidence.pyannote import (
    PyannoteEvidenceConfig,
    PyannoteSegment,
    _rasterize_segments,
    compute_pyannote_evidence,
)


def test_rasterize_segments_pads_and_clamps_to_canonical_grid() -> None:
    tb = TimeBase.canonical(n_frames=10)
    track = _rasterize_segments(
        segments=[PyannoteSegment(start=0.02, end=0.05, label="SPEAKER_00")],
        timebase=tb,
        pad_seconds=0.02,
        track_name="pyannote_speech",
    )

    assert track.values.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_empty_pyannote_output_produces_zero_track(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "empty.wav"
    audio.write_bytes(b"RIFF")
    tb = TimeBase.canonical(n_frames=8)
    cfg = PyannoteEvidenceConfig(enabled=True, hf_token="token")

    monkeypatch.setattr("dyana.evidence.pyannote._load_segments_from_pyannote", lambda *_args, **_kwargs: [])

    result = compute_pyannote_evidence(audio, tb, cfg, cache_dir=tmp_path)

    assert result.status == "ok"
    assert "pyannote_speech" in result.bundle.tracks
    assert np.count_nonzero(result.bundle.get("pyannote_speech").values) == 0


def test_anonymous_speaker_tracks_are_deterministic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "speakers.wav"
    audio.write_bytes(b"RIFF")
    tb = TimeBase.canonical(n_frames=20)
    cfg = PyannoteEvidenceConfig(enabled=True, hf_token="token", max_speakers=2)
    segments = [
        PyannoteSegment(start=0.0, end=0.08, label="speaker_b"),
        PyannoteSegment(start=0.10, end=0.18, label="speaker_a"),
    ]

    monkeypatch.setattr("dyana.evidence.pyannote._load_segments_from_pyannote", lambda *_args, **_kwargs: segments)

    result = compute_pyannote_evidence(audio, tb, cfg, cache_dir=tmp_path)

    assert result.speaker_labels == ("speaker_a", "speaker_b")
    assert set(result.bundle.tracks) == {"pyannote_speech", "pyannote_speaker_0", "pyannote_speaker_1"}


def test_pyannote_unavailable_disabled_does_not_crash(tmp_path: Path) -> None:
    audio = tmp_path / "disabled.wav"
    audio.write_bytes(b"RIFF")

    result = compute_pyannote_evidence(audio, TimeBase.canonical(n_frames=4), PyannoteEvidenceConfig(enabled=False))

    assert result.status == "disabled"
    assert result.bundle.tracks == {}


def test_pyannote_unavailable_enabled_degrades_in_run_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "runmode.wav"
    audio.write_bytes(b"RIFF")
    tb = TimeBase.canonical(n_frames=5)
    cfg = PyannoteEvidenceConfig(enabled=True, hf_token="token")

    monkeypatch.setattr(
        "dyana.evidence.pyannote._load_segments_from_pyannote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OptionalBackendUnavailableError("pyannote missing")),
    )

    result = compute_pyannote_evidence(audio, tb, cfg, error_mode="run")

    assert result.status == "failed"
    assert result.message == "pyannote missing"
    assert result.bundle.tracks == {}


def test_pyannote_unavailable_enabled_raises_in_debug_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "debugmode.wav"
    audio.write_bytes(b"RIFF")
    tb = TimeBase.canonical(n_frames=5)
    cfg = PyannoteEvidenceConfig(enabled=True, hf_token="token")

    monkeypatch.setattr(
        "dyana.evidence.pyannote._load_segments_from_pyannote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OptionalBackendUnavailableError("pyannote missing")),
    )

    with pytest.raises(OptionalBackendUnavailableError, match="pyannote missing"):
        compute_pyannote_evidence(audio, tb, cfg, error_mode="debug")


def test_missing_token_raises_clear_configuration_error(tmp_path: Path) -> None:
    audio = tmp_path / "token.wav"
    audio.write_bytes(b"RIFF")

    with pytest.raises(BackendConfigurationError, match="Hugging Face token"):
        compute_pyannote_evidence(
            audio,
            TimeBase.canonical(n_frames=5),
            PyannoteEvidenceConfig(enabled=True, hf_token=None, model_name="pyannote/model"),
            error_mode="debug",
        )
