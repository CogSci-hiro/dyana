from pathlib import Path

import numpy as np
import soundfile as sf

import dyana.evidence.vad as vad_module
from dyana.evidence.prosody import compute_voiced_soft_track
from dyana.evidence.vad import compute_webrtc_vad_soft_track


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data, sr)


def test_vad_fallback_without_webrtcvad(monkeypatch, tmp_path: Path) -> None:
    sr = 16000
    tone = np.concatenate([np.zeros(sr // 4), 0.04 * np.ones(sr // 2), np.zeros(sr // 4)]).astype(np.float32)
    path = tmp_path / "fallback.wav"
    _write_wav(path, tone, sr)
    monkeypatch.setattr(vad_module, "webrtcvad", None)

    track = compute_webrtc_vad_soft_track(path)

    assert track.name == "vad"
    assert track.semantics == "probability"
    assert float(track.values.min()) >= 0.0
    assert float(track.values.max()) <= 1.0
    assert float(track.values.max()) > float(track.values.min())


def test_voiced_soft_works_without_webrtcvad(monkeypatch, tmp_path: Path) -> None:
    sr = 16000
    tone = np.concatenate([np.zeros(sr // 3), 0.03 * np.ones(sr // 3), np.zeros(sr // 3)]).astype(np.float32)
    path = tmp_path / "voiced_fallback.wav"
    _write_wav(path, tone, sr)
    monkeypatch.setattr(vad_module, "webrtcvad", None)

    track = compute_voiced_soft_track(path)

    assert track.name == "voiced_soft"
    assert track.semantics == "probability"
    assert float(track.values.min()) >= 0.0
    assert float(track.values.max()) <= 1.0
