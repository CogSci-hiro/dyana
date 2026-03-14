from __future__ import annotations

from typing import Any, get_type_hints

from dyana.core.contracts.transcript import Token, TranscriptBundle


def test_transcript_contract_dataclasses_can_be_constructed() -> None:
    bundle = TranscriptBundle(tokens=[Token(text="hello", speaker="A")], language="en", metadata={"source": "unit"})

    assert bundle.tokens[0].text == "hello"
    assert bundle.metadata == {"source": "unit"}


def test_transcript_contract_type_hints() -> None:
    hints = get_type_hints(TranscriptBundle)

    assert hints["tokens"] == list[Token]
    assert hints["language"] == str | None
    assert hints["metadata"] == dict[str, Any] | None
