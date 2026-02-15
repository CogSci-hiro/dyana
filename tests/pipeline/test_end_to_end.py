import importlib.util
import numpy as np
import soundfile as sf
from pathlib import Path
import pytest

pytest.importorskip("webrtcvad")
from dyana.pipeline.run_pipeline import run_pipeline


def _make_audio(path: Path) -> None:
    sr = 16000
    t = np.arange(0, sr * 1.0) / sr
    sig = np.concatenate([np.zeros(sr // 4), 0.05 * np.sin(2 * np.pi * 200 * t[: sr // 2]), np.zeros(sr // 4)])
    sf.write(path, sig, sr)


def test_pipeline_runs_and_writes_outputs(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    _make_audio(audio)
    out_dir = tmp_path / "out1"
    summary = run_pipeline(audio, out_dir=out_dir)
    assert (out_dir / "a.TextGrid").exists()
    assert (out_dir / "evidence" / "a_vad_soft.npz").exists()
    assert summary["n_frames"] > 0


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    audio = tmp_path / "b.wav"
    _make_audio(audio)
    out1 = tmp_path / "out_a"
    out2 = tmp_path / "out_b"
    run_pipeline(audio, out_dir=out1)
    run_pipeline(audio, out_dir=out2)
    states1 = np.load(out1 / "decode" / "b_states.npy", allow_pickle=True)
    states2 = np.load(out2 / "decode" / "b_states.npy", allow_pickle=True)
    assert np.array_equal(states1, states2)
