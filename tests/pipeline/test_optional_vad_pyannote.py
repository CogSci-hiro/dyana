from pathlib import Path

import numpy as np
import pytest

from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence.pyannote import PyannoteEvidenceConfig, PyannoteEvidenceResult
from dyana.pipeline.run_pipeline import run_pipeline


sf = pytest.importorskip("soundfile")


def _make_audio(path: Path) -> None:
    sr = 16000
    signal = np.concatenate([np.zeros(sr // 4), 0.05 * np.ones(sr // 2), np.zeros(sr // 4)]).astype(np.float32)
    sf.write(path, signal, sr)


def test_pipeline_runs_without_webrtc_when_vad_backend_none(tmp_path: Path) -> None:
    audio = tmp_path / "novad.wav"
    _make_audio(audio)

    summary = run_pipeline(audio, out_dir=tmp_path / "out", vad_backend="none")

    assert summary["n_frames"] > 0
    assert summary["vad_backend"] == "none"
    assert (tmp_path / "out" / "decode" / "novad_states.npy").exists()


def test_pipeline_emits_pyannote_none_diagnostics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio = tmp_path / "pyannote.wav"
    _make_audio(audio)

    def _fake_pyannote(*_args, **_kwargs) -> PyannoteEvidenceResult:
        tb = TimeBase.canonical(n_frames=100)
        bundle = EvidenceBundle(timebase=tb)
        values = np.zeros(100, dtype=np.float32)
        values[20:80] = 1.0
        bundle.add_track("pyannote_speech", EvidenceTrack("pyannote_speech", tb, values, "probability"))
        return PyannoteEvidenceResult(bundle=bundle, status="ok")

    monkeypatch.setattr("dyana.pipeline.run_pipeline.compute_pyannote_evidence", _fake_pyannote)

    summary = run_pipeline(
        audio,
        out_dir=tmp_path / "pyannote_out",
        vad_backend="none",
        pyannote_config=PyannoteEvidenceConfig(enabled=True, hf_token="token"),
        profile="recall-first",
    )

    diagnostics = summary["diagnostics"]
    assert "pyannote_speech_but_decoded_none_seconds" in diagnostics
    assert diagnostics["pyannote_status"] == "ok"
