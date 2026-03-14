from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from dyana import align, annotate, decode_structure
from dyana.api.align import align as api_align
from dyana.api.annotate import annotate as api_annotate
from dyana.api.structure import decode_structure as api_decode_structure
from dyana.api.types import Alignment, AnnotationResult, ConversationalState, IPU, Transcript, Word


def test_root_package_re_exports_public_api() -> None:
    assert annotate is api_annotate
    assert align is api_align
    assert decode_structure is api_decode_structure


def test_annotate_signature_types() -> None:
    hints = get_type_hints(annotate)

    assert hints["audio"] == str | Path
    assert hints["language"] == str | None
    assert hints["diarize"] is bool
    assert hints["align"] is bool
    assert hints["return"] is AnnotationResult


def test_align_signature_types() -> None:
    hints = get_type_hints(align)

    assert hints["audio"] == str | Path
    assert hints["transcript"] == Transcript | str | Path
    assert hints["language"] == str | None
    assert hints["return"] is Alignment


def test_decode_structure_signature_types() -> None:
    hints = get_type_hints(decode_structure)

    assert hints["audio"] == str | Path
    assert hints["speakers"] == tuple[str, str]
    assert hints["return"] == tuple[list[IPU], list[ConversationalState]]


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (annotate, (Path("sample.wav"),), "Transcription step not implemented yet"),
        (align, (Path("sample.wav"), Transcript(words=[Word(text="hi", start=0.0, end=0.1)])), "Pipeline not implemented yet"),
        (decode_structure, (Path("sample.wav"),), "Pipeline not implemented yet"),
    ],
)
def test_public_api_functions_are_stubbed(
    function,
    args: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(NotImplementedError, match=message):
        function(*args)
