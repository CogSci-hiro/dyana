from pathlib import Path

import numpy as np
import pytest

from dyana.io.audio import detect_channel_similarity, load_audio_mono


sf = pytest.importorskip("soundfile")


def test_load_multichannel_mixes_down(tmp_path: Path) -> None:
    sr = 8000
    data = np.stack([np.ones(sr), np.zeros(sr)], axis=1)  # stereo
    path = tmp_path / "stereo.wav"
    sf.write(path, data, sr, subtype="FLOAT")

    mono, out_sr = load_audio_mono(path)
    assert out_sr == sr
    assert mono.shape[0] == sr
    assert np.isclose(mono.mean(), 0.5, atol=1e-3)

    ch0, _ = load_audio_mono(path, channel=0)
    assert np.allclose(ch0, 1.0, atol=1e-6)


def test_detect_channel_similarity_same_signal(tmp_path: Path) -> None:
    sr = 16000
    tone = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    stereo = np.stack([tone, 0.5 * tone], axis=1)
    path = tmp_path / "same.wav"
    sf.write(path, stereo, sr, subtype="FLOAT")

    label, correlation = detect_channel_similarity(str(path))

    assert label == "same"
    assert correlation > 0.98


def test_detect_channel_similarity_different_signal(tmp_path: Path) -> None:
    sr = 16000
    left = np.concatenate([np.ones(sr // 2), np.zeros(sr // 2)]).astype(np.float32)
    right = np.concatenate([np.zeros(sr // 2), np.ones(sr // 2)]).astype(np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "different.wav"
    sf.write(path, stereo, sr, subtype="FLOAT")

    label, correlation = detect_channel_similarity(str(path))

    assert label == "different"
    assert correlation < 0.98


def test_detect_channel_similarity_requires_stereo(tmp_path: Path) -> None:
    sr = 16000
    mono = np.zeros(sr, dtype=np.float32)
    path = tmp_path / "mono.wav"
    sf.write(path, mono, sr, subtype="FLOAT")

    with pytest.raises(ValueError, match="Expected stereo audio"):
        detect_channel_similarity(str(path))
