import importlib.util
import numpy as np
import soundfile as sf
from pathlib import Path
import pytest

pytest.importorskip("webrtcvad")
from dyana.evidence.prosody import compute_voiced_soft_track, compute_energy_slope_prosody_track


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data, sr)


def test_voiced_soft_track_uses_vad(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(0, sr) / sr
    tone = 0.05 * np.sin(2 * np.pi * 120 * t)
    path = tmp_path / "tone_voiced.wav"
    _write_wav(path, tone, sr)

    track = compute_voiced_soft_track(path)
    assert track.semantics == "probability"
    assert track.values.min() >= 0.0 and track.values.max() <= 1.0


def test_energy_slope_prosody_matches_energy_slope(tmp_path: Path) -> None:
    sr = 16000
    low = np.zeros(sr // 2)
    high = np.ones(sr // 2)
    step = np.concatenate([low, high])
    path = tmp_path / "step_prosody.wav"
    _write_wav(path, step, sr)

    track = compute_energy_slope_prosody_track(path)
    assert track.name == "energy_slope"
    assert track.values.max() > 0.05
