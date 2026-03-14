from __future__ import annotations

from typing import get_type_hints

import numpy as np

from dyana.core.contracts.evidence import EvidenceBundle, EvidenceTrack


def test_evidence_contract_dataclasses_can_be_constructed() -> None:
    values = np.array([0.1, 0.2], dtype=np.float32)
    track = EvidenceTrack(name="energy", values=values, frame_hop=0.01, start_time=0.0)
    bundle = EvidenceBundle(tracks={"energy": track}, duration=0.02)

    assert bundle.tracks["energy"].values is values
    assert bundle.duration == 0.02


def test_evidence_contract_type_hints() -> None:
    hints = get_type_hints(EvidenceBundle)

    assert hints["tracks"] == dict[str, EvidenceTrack]
    assert hints["duration"] is float
