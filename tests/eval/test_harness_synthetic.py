import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

webrtcvad = pytest.importorskip("webrtcvad")

from dyana.eval.harness import evaluate_manifest
from dyana.eval.scorecard import write_scorecard


def _make_audio(path: Path) -> None:
    sr = 16000
    t = np.arange(0, sr * 0.5) / sr
    sig = np.concatenate([np.zeros(sr // 4), 0.05 * np.sin(2 * np.pi * 220 * t[: sr // 4]), np.zeros(sr // 4)])
    sf.write(path, sig, sr)


def test_harness_runs_and_writes_scorecard(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    _make_audio(audio)
    ref_states = ["SIL"] * 50
    ref_path = tmp_path / "ref.npy"
    np.save(ref_path, np.array(ref_states, dtype=object))

    manifest = [
        {"id": "a", "audio_path": str(audio), "ref_path": str(ref_path), "tier": "synthetic"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    results = evaluate_manifest(manifest_path, out_dir=tmp_path / "out")
    write_scorecard(results, tmp_path / "out")
    assert (tmp_path / "out" / "scorecard.json").exists()
    assert results[0]["id"] == "a"
