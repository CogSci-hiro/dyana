import numpy as np

from dyana.core.timebase import TimeBase
from dyana.evidence.stereo import (
    compute_channel_energy,
    compute_cross_channel_correlation,
    compute_energy_ratio,
    compute_stereo_evidence,
)


def test_energy_symmetry_identical_channels_ratio_is_zero() -> None:
    frame_hop = 16
    frames = 20
    t = np.arange(frame_hop * frames, dtype=np.float32)
    signal = np.sin(2.0 * np.pi * t / frame_hop).astype(np.float32)
    stereo = np.stack([signal, signal], axis=1)

    left_energy, right_energy = compute_channel_energy(stereo, frame_hop)
    ratio = compute_energy_ratio(left_energy, right_energy)

    assert np.allclose(left_energy, right_energy, atol=1e-6)
    assert np.allclose(ratio, 0.0, atol=1e-6)


def test_channel_dominance_left_louder_yields_positive_ratio() -> None:
    frame_hop = 16
    frames = 20
    t = np.arange(frame_hop * frames, dtype=np.float32)
    left = 0.8 * np.sin(2.0 * np.pi * t / frame_hop)
    right = 0.2 * np.sin(2.0 * np.pi * t / frame_hop)
    stereo = np.stack([left, right], axis=1).astype(np.float32)

    left_energy, right_energy = compute_channel_energy(stereo, frame_hop)
    ratio = compute_energy_ratio(left_energy, right_energy)

    assert np.all(left_energy > right_energy)
    assert np.all(ratio > 0.0)


def test_cross_channel_correlation_identical_signals_is_one() -> None:
    frame_hop = 32
    frames = 12
    t = np.arange(frame_hop * frames, dtype=np.float32)
    signal = np.sin(2.0 * np.pi * t / 10.0).astype(np.float32)
    stereo = np.stack([signal, signal], axis=1)

    corr = compute_cross_channel_correlation(stereo, frame_hop)

    assert np.allclose(corr, 1.0, atol=1e-6)


def test_cross_channel_correlation_independent_noise_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    frame_hop = 128
    frames = 64
    left = rng.standard_normal(frame_hop * frames).astype(np.float32)
    right = rng.standard_normal(frame_hop * frames).astype(np.float32)
    stereo = np.stack([left, right], axis=1)

    corr = compute_cross_channel_correlation(stereo, frame_hop)

    assert abs(float(corr.mean())) < 0.1


def test_compute_stereo_evidence_returns_expected_tracks() -> None:
    sample_rate = 16000
    duration_s = 0.2
    samples = int(sample_rate * duration_s)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    left = 0.7 * np.sin(2.0 * np.pi * 220.0 * t)
    right = 0.3 * np.sin(2.0 * np.pi * 220.0 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    timebase = TimeBase.canonical(n_frames=int(round(duration_s / TimeBase.canonical().hop_s)))

    bundle = compute_stereo_evidence((stereo, sample_rate), timebase)

    assert set(bundle.keys()) == {
        "stereo_energy_left",
        "stereo_energy_right",
        "stereo_ratio",
        "stereo_corr",
    }
    assert bundle.get("stereo_ratio") is not None
    assert bundle.get("stereo_corr") is not None
    assert bundle.get("stereo_ratio").values.shape[0] == timebase.n_frames
    assert bundle.get("stereo_corr").values.shape[0] == timebase.n_frames
