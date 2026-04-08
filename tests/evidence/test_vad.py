import importlib.util
import numpy as np
from pathlib import Path
import pytest

from dyana.evidence.vad import compute_webrtc_vad_soft_track


if importlib.util.find_spec("soundfile") is not None:
    import soundfile as sf
else:
    sf = None

pytestmark = pytest.mark.skipif(
    sf is None or importlib.util.find_spec("webrtcvad") is None,
    reason="soundfile and webrtcvad are required for this test module",
)


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
