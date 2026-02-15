"""Artifact writing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dyana.evidence.base import EvidenceTrack


def save_evidence_track(track: EvidenceTrack, path: Path) -> None:
    arrays = {"values": track.values}
    meta = {
        "name": track.name,
        "semantics": track.semantics,
        "timebase": {"hop_s": track.timebase.hop_s, "n_frames": track.timebase.n_frames},
        "metadata": dict(track.metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata=json.dumps(meta))


def save_states(states: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(states))


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def dump_diagnostics(out_dir: Path, stem: str, diagnostics: Mapping[str, Any]) -> Path:
    """Write deterministic diagnostics JSON under decode artifacts."""

    path = out_dir / "decode" / f"{stem}_diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(diagnostics), indent=2, sort_keys=True))
    return path
