"""TextGrid export for structured DYANA annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dyana.api.types import AnnotationResult


def export_textgrid(annotation_result: AnnotationResult, path: Path) -> None:
    word_intervals = [
        (word.start, word.end, word.text)
        for word in annotation_result.transcript.words
    ]
    phoneme_intervals = [
        (phoneme.start, phoneme.end, phoneme.symbol)
        for phoneme in (annotation_result.alignment.phonemes or [])
    ]
    ipu_intervals = [
        (ipu.start, ipu.end, ipu.speaker)
        for ipu in annotation_result.ipus
    ]
    state_intervals = [
        (state.start, state.end, state.label)
        for state in annotation_result.states
    ]

    xmax = max(
        [0.0]
        + [end for _, end, _ in word_intervals]
        + [end for _, end, _ in phoneme_intervals]
        + [end for _, end, _ in ipu_intervals]
        + [end for _, end, _ in state_intervals]
    )
    tiers = [
        _tier_block(1, "words", word_intervals, xmax=xmax),
        _tier_block(2, "phonemes", phoneme_intervals, xmax=xmax),
        _tier_block(3, "ipus", ipu_intervals, xmax=xmax),
        _tier_block(4, "states", state_intervals, xmax=xmax),
    ]
    header = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {_format_number(0.0)}",
        f"xmax = {_format_number(xmax)}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + [line for tier in tiers for line in tier]))


def _tier_block(index: int, name: str, intervals: Iterable[tuple[float, float, str]], *, xmax: float) -> list[str]:
    filled = _fill_gaps(list(intervals), xmax=xmax)
    lines = [
        f"    item [{index}]:",
        '        class = "IntervalTier"',
        f'        name = "{name}"',
        f"        xmin = {_format_number(0.0)}",
        f"        xmax = {_format_number(xmax)}",
        f"        intervals: size = {len(filled)}",
    ]
    for offset, (start, end, label) in enumerate(filled, start=1):
        lines.extend(
            [
                f"        intervals [{offset}]:",
                f"            xmin = {_format_number(start)}",
                f"            xmax = {_format_number(end)}",
                f'            text = "{label}"',
            ]
        )
    return lines


def _fill_gaps(intervals: list[tuple[float, float, str]], *, xmax: float) -> list[tuple[float, float, str]]:
    if not intervals:
        return [(0.0, xmax, "")]

    ordered = sorted(intervals, key=lambda item: (item[0], item[1], item[2]))
    filled: list[tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, label in ordered:
        if start > cursor:
            filled.append((cursor, start, ""))
        filled.append((start, end, label))
        cursor = end
    if cursor < xmax:
        filled.append((cursor, xmax, ""))
    return filled


def _format_number(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")
