from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dyana.core.contracts.alignment import AlignmentBundle, PhonemeInterval, WordInterval


def test_alignment_contract_dataclasses_can_be_constructed() -> None:
    word = WordInterval(text="hello", start=0.0, end=0.5, speaker="A")
    phoneme = PhonemeInterval(symbol="HH", start=0.0, end=0.1)
    bundle = AlignmentBundle(words=[word], phonemes=[phoneme])

    assert bundle.words[0].speaker == "A"
    assert bundle.phonemes is not None
    assert bundle.phonemes[0].symbol == "HH"


def test_alignment_contract_dataclasses_are_frozen() -> None:
    interval = PhonemeInterval(symbol="HH", start=0.0, end=0.1)

    with pytest.raises(FrozenInstanceError):
        interval.symbol = "AH"  # type: ignore[misc]
