from __future__ import annotations

from typing import get_type_hints

import numpy as np

from dyana.core.contracts.alignment import AlignmentBundle, WordInterval
from dyana.core.contracts.decoder import DecodeResult, DecoderInput, IPUInterval, StateInterval
from dyana.core.contracts.evidence import EvidenceBundle, EvidenceTrack


def test_decoder_contract_dataclasses_can_be_constructed() -> None:
    alignment = AlignmentBundle(words=[WordInterval(text="hello", start=0.0, end=0.5, speaker="A")], phonemes=None)
    track = EvidenceTrack(
        name="energy",
        values=np.array([0.1, 0.2], dtype=np.float32),
        frame_hop=0.01,
        start_time=0.0,
    )
    evidence = EvidenceBundle(tracks={"energy": track}, duration=0.02)
    decoder_input = DecoderInput(alignment=alignment, evidence=evidence, speakers=("A", "B"))
    result = DecodeResult(
        states=[StateInterval(label="floor_hold", start=0.0, end=0.5)],
        ipus=[IPUInterval(speaker="A", start=0.0, end=0.5)],
    )

    assert decoder_input.speakers == ("A", "B")
    assert result.states[0].label == "floor_hold"


def test_decoder_contract_type_hints() -> None:
    hints = get_type_hints(DecoderInput)

    assert hints["speakers"] == tuple[str, ...]
