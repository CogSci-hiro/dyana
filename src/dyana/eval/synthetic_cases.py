"""Deterministic synthetic evaluation cases."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf


SAMPLE_RATE: int = 16000
LEAKAGE_STRESS_ID: str = "leakage_stress"


def _tone(freq_hz: float, frames: int, amplitude: float = 0.06) -> np.ndarray:
    times = np.arange(frames, dtype=np.float32) / float(SAMPLE_RATE)
    return amplitude * np.sin(2.0 * np.pi * freq_hz * times)


def _build_leakage_stress_audio() -> tuple[np.ndarray, List[str]]:
    segment_frames = int(0.5 * SAMPLE_RATE)
    silence = np.zeros(segment_frames, dtype=np.float32)
    tone_a = _tone(220.0, segment_frames)
    tone_b = _tone(330.0, segment_frames)
    leak = _tone(220.0, segment_frames, amplitude=0.05)

    left = np.concatenate([silence, tone_a, silence, leak, silence, tone_b, silence])
    right = np.concatenate([silence, 0.03 * tone_a, silence, 0.01 * leak, silence, tone_b, silence])
    stereo = np.stack([left, right], axis=1)

    ref_states: List[str] = []
    for label in ("SIL", "A", "SIL", "LEAK", "SIL", "B", "SIL"):
        ref_states.extend([label] * 50)
    return stereo, ref_states


def materialize_synthetic_case(
    item: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Materialize a synthetic evaluation item as audio + reference files.

    Parameters
    ----------
    item
        Manifest item with ``scenario`` key.
    out_dir
        Directory to write generated files.
    """

    scenario = str(item.get("scenario", "")).strip()
    if scenario != LEAKAGE_STRESS_ID:
        raise ValueError(f"Unsupported synthetic scenario '{scenario}'.")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(item.get("id", "synthetic"))
    audio_path = out_dir / f"{stem}.wav"
    ref_path = out_dir / f"{stem}_ref.npy"

    stereo, ref_states = _build_leakage_stress_audio()
    sf.write(audio_path, stereo, SAMPLE_RATE)
    np.save(ref_path, np.array(ref_states, dtype=object))

    resolved = dict(item)
    resolved["audio_path"] = str(audio_path)
    resolved["ref_path"] = str(ref_path)
    return resolved
