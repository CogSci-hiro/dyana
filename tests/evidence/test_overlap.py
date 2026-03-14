from __future__ import annotations

import numpy as np

from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.overlap import OVERLAP_PROXY_TRACK_NAME, compute_overlap_proxy_tracker


def test_overlap_proxy_tracker_multiplies_speaker_activity() -> None:
    tb = TimeBase.canonical(n_frames=3)
    diar_a = EvidenceTrack(
        name="diar_a",
        timebase=tb,
        values=np.array([0.2, 0.5, 1.0], dtype=np.float32),
        semantics="probability",
    )
    diar_b = EvidenceTrack(
        name="diar_b",
        timebase=tb,
        values=np.array([0.5, 0.5, 0.25], dtype=np.float32),
        semantics="probability",
    )

    track = compute_overlap_proxy_tracker(diar_a, diar_b)

    assert track.name == OVERLAP_PROXY_TRACK_NAME
    np.testing.assert_allclose(track.values, np.array([0.1, 0.25, 0.25], dtype=np.float32))
