import numpy as np
import soundfile as sf
from pathlib import Path

from dyana.evidence.vad import compute_webrtc_vad_soft_track


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data, sr)


def test_webrtc_vad_soft_outputs_probabilities(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(0, sr) / sr
    tone = 0.02 * np.sin(2 * np.pi * 200 * t)
    path = tmp_path / "tone.wav"
    _write_wav(path, tone, sr)

    track = compute_webrtc_vad_soft_track(path, subframe_ms=5)
    vals = track.values
    assert track.semantics == "probability"
    assert vals.dtype.kind == "f"
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert len(np.unique(vals)) > 2  # not purely binary
