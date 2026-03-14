"""Overlap proxy evidence helpers."""

from __future__ import annotations

import numpy as np

from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceTrack


OVERLAP_PROXY_TRACK_NAME: str = "overlap_proxy"


def _as_probability(track: EvidenceTrack, *, label: str) -> np.ndarray:
    values = np.asarray(track.values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"{label} must be 1-D, got shape {values.shape}.")
    if track.semantics == "probability":
        return np.clip(values, 0.0, 1.0)
    if track.semantics == "logit":
        return 1.0 / (1.0 + np.exp(-values))
    raise ValueError(f"{label} must use probability or logit semantics, got {track.semantics}.")


def compute_overlap_proxy_tracker(
    diar_a: EvidenceTrack,
    diar_b: EvidenceTrack,
    *,
    name: str = OVERLAP_PROXY_TRACK_NAME,
) -> EvidenceTrack:
    """
    Build a simple overlap proxy from speaker-A and speaker-B activity tracks.

    The proxy is intentionally lightweight: it treats simultaneous activity as
    overlap evidence by multiplying the per-frame speaker probabilities.
    """

    if diar_a.T != diar_b.T:
        raise ValueError(f"diar_a length {diar_a.T} does not match diar_b length {diar_b.T}.")
    if abs(diar_a.timebase.hop_s - diar_b.timebase.hop_s) > 1e-12:
        raise ValueError(
            f"diar_a hop {diar_a.timebase.hop_s} does not match diar_b hop {diar_b.timebase.hop_s}."
        )

    p_a = _as_probability(diar_a, label="diar_a")
    p_b = _as_probability(diar_b, label="diar_b")
    values = (p_a * p_b).astype(np.float32, copy=False)
    timebase = TimeBase.from_hop_seconds(diar_a.timebase.hop_s, n_frames=diar_a.T)
    return EvidenceTrack(
        name=name,
        timebase=timebase,
        values=values,
        semantics="probability",
        metadata={"source": "derived", "kind": "overlap_proxy"},
    )
