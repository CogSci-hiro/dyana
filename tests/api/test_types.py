from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dyana.api.types import (
    Alignment,
    AnnotationResult,
    ConversationalState,
    Diagnostics,
    IPU,
    Phoneme,
    Transcript,
    Word,
)


def test_public_type_dataclasses_can_be_constructed() -> None:
    word = Word(text="hello", start=0.0, end=0.5, speaker="A", confidence=0.9)
    transcript = Transcript(words=[word], language="en")
    alignment = Alignment(words=[word], phonemes=[Phoneme(symbol="HH", start=0.0, end=0.1)])
    ipu = IPU(speaker="A", start=0.0, end=1.0)
    state = ConversationalState(label="floor_hold", start=0.0, end=1.0)
    diagnostics = Diagnostics(metrics={"coverage": 1.0})

    result = AnnotationResult(
        transcript=transcript,
        alignment=alignment,
        ipus=[ipu],
        states=[state],
        diagnostics=diagnostics,
    )

    assert result.transcript.language == "en"
    assert result.alignment.words[0].text == "hello"
    assert result.diagnostics.metrics["coverage"] == pytest.approx(1.0)


def test_public_type_dataclasses_are_frozen() -> None:
    word = Word(text="hello", start=0.0, end=0.5)

    with pytest.raises(FrozenInstanceError):
        word.text = "updated"  # type: ignore[misc]
