import numpy as np
import soundfile as sf
from pathlib import Path

from dyana.evidence.energy import (
    compute_energy_rms_track,
    compute_energy_smooth_track,
    compute_energy_slope_track,
)


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data, sr)


def test_energy_rms_silence_vs_tone(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(0, sr) / sr
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    silence = np.zeros_like(sine)

    sine_path = tmp_path / "sine.wav"
    sil_path = tmp_path / "sil.wav"
    _write_wav(sine_path, sine, sr)
    _write_wav(sil_path, silence, sr)

    sine_track = compute_energy_rms_track(sine_path)
    sil_track = compute_energy_rms_track(sil_path)

    assert sine_track.timebase.hop_s == 0.01
    assert sil_track.timebase.hop_s == 0.01
    assert sil_track.values.max() < 1e-4
    assert sine_track.values.mean() > sil_track.values.mean()


def test_energy_smoothing_softens_step(tmp_path: Path) -> None:
    sr = 16000
    low = np.zeros(sr // 2)
    high = np.ones(sr // 2)
    step = np.concatenate([low, high])
    path = tmp_path / "step.wav"
    _write_wav(path, step, sr)

    smooth_short = compute_energy_smooth_track(path, smooth_ms=20.0)
    smooth_long = compute_energy_smooth_track(path, smooth_ms=120.0)

    mid = len(smooth_short.values) // 2
    jump_short = smooth_short.values[mid] - smooth_short.values[mid - 1]
    jump_long = smooth_long.values[mid] - smooth_long.values[mid - 1]
    assert abs(jump_long) < abs(jump_short)


def test_energy_slope_detects_rise(tmp_path: Path) -> None:
    sr = 16000
    low = np.zeros(sr // 2)
    high = np.ones(sr // 2)
    step = np.concatenate([low, high])
    path = tmp_path / "step2.wav"
    _write_wav(path, step, sr)

    slope = compute_energy_slope_track(path)
    assert slope.values.max() > 0.05
    # After rise, slope near zero compared to peak
    assert np.abs(slope.values[-10:]).mean() < 0.1 * slope.values.max()
