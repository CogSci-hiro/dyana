from pathlib import Path

import numpy as np
import soundfile as sf

from dyana.io.audio import load_audio_mono


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
