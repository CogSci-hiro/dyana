from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase
from dyana.decode import decoder, fusion
from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence.leakage import compute_leakage_likelihood
from dyana.evidence.synthetic import make_timebase, make_vad_track


def _write_audio(path: Path, data: np.ndarray, sample_rate: int = 16000) -> None:
    sf.write(path, data, sample_rate, subtype="FLOAT")


def _sine(sample_rate: int, seconds: float, freq_hz: float, amp: float) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def test_leakage_track_is_deterministic(tmp_path: Path) -> None:
    sample_rate = 16000
    left = _sine(sample_rate, 1.0, 200.0, 0.8)
    right = _sine(sample_rate, 1.0, 200.0, 0.2)
    stereo = np.stack([left, right], axis=1)
    audio_path = tmp_path / "dominated.wav"
    cache_dir = tmp_path / "cache"
    _write_audio(audio_path, stereo, sample_rate)

    track_a = compute_leakage_likelihood(audio_path, cache_dir=cache_dir)
    track_b = compute_leakage_likelihood(audio_path, cache_dir=cache_dir)

    assert track_a.timebase.hop_s == CANONICAL_HOP_SECONDS
    assert track_b.timebase.hop_s == CANONICAL_HOP_SECONDS
    assert np.array_equal(track_a.values, track_b.values)


def test_leakage_high_when_one_channel_dominates_and_spectra_similar(tmp_path: Path) -> None:
    sample_rate = 16000
    left = _sine(sample_rate, 1.0, 200.0, 0.8)
    right = _sine(sample_rate, 1.0, 200.0, 0.2)
    stereo = np.stack([left, right], axis=1)
    dominated_path = tmp_path / "dominated.wav"
    _write_audio(dominated_path, stereo, sample_rate)

    silence = np.zeros((sample_rate, 2), dtype=np.float32)
    silence_path = tmp_path / "silence.wav"
    _write_audio(silence_path, silence, sample_rate)

    dominated = compute_leakage_likelihood(dominated_path)
    baseline = compute_leakage_likelihood(silence_path)
    assert float(np.mean(dominated.values)) > 0.2
    assert float(np.mean(dominated.values)) > float(np.mean(baseline.values))


def test_leakage_near_zero_when_channels_different(tmp_path: Path) -> None:
    sample_rate = 16000
    left = _sine(sample_rate, 1.0, 200.0, 0.6)
    right = _sine(sample_rate, 1.0, 900.0, 0.6)
    stereo = np.stack([left, right], axis=1)
    audio_path = tmp_path / "different.wav"
    _write_audio(audio_path, stereo, sample_rate)

    track = compute_leakage_likelihood(audio_path)
    assert float(np.mean(track.values)) < 0.1


def test_leakage_shape_and_bounds(tmp_path: Path) -> None:
    sample_rate = 16000
    left = _sine(sample_rate, 1.0, 300.0, 0.5)
    right = _sine(sample_rate, 1.0, 300.0, 0.3)
    audio_path = tmp_path / "shape.wav"
    _write_audio(audio_path, np.stack([left, right], axis=1), sample_rate)

    track = compute_leakage_likelihood(audio_path)
    assert track.values.ndim == 1
    assert track.values.dtype.kind == "f"
    assert np.isfinite(track.values).all()
    assert np.all(track.values >= 0.0)
    assert np.all(track.values <= 1.0)
    assert track.timebase.n_frames == track.values.shape[0]


def test_mono_input_raises(tmp_path: Path) -> None:
    sample_rate = 16000
    mono = _sine(sample_rate, 1.0, 200.0, 0.5)
    path = tmp_path / "mono.wav"
    _write_audio(path, mono, sample_rate)
    with pytest.raises(ValueError, match="requires stereo"):
        compute_leakage_likelihood(path)


def test_decoder_does_not_crash_without_leakage_track() -> None:
    tb = make_timebase(100)
    vad = make_vad_track(tb, [(20, 80)], p_speech=0.9, p_sil=0.1)
    bundle = EvidenceBundle(timebase=tb)
    bundle.add_track("vad", vad)
    scores = fusion.fuse_bundle_to_scores(bundle)
    path = decoder.decode_with_constraints(scores)
    assert len(path) == 100
