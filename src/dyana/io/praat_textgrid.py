"""Praat TextGrid I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

from dyana.decode.ipu import Segment


def _format_number(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _fill_with_silence(segments: List[Segment], *, xmax: float, silence_label: str) -> List[Segment]:
    if xmax <= 0:
        return [Segment(start_time=0.0, end_time=0.0, label=silence_label)]

    filled: List[Segment] = []
    cursor = 0.0
    for seg in segments:
        if seg.start_time > cursor:
            filled.append(Segment(start_time=cursor, end_time=seg.start_time, label=silence_label))
        filled.append(seg)
        cursor = seg.end_time
    if cursor < xmax:
        filled.append(Segment(start_time=cursor, end_time=xmax, label=silence_label))
    if not filled:
        filled.append(Segment(start_time=0.0, end_time=xmax, label=silence_label))
    return filled


def _tier_block(index: int, name: str, segments: Iterable[Segment], *, xmax: float, silence_label: str) -> List[str]:
    seg_list = list(segments)
    seg_list = _fill_with_silence(seg_list, xmax=xmax, silence_label=silence_label)
    lines = [
        f'    item [{index}]:',
        '        class = "IntervalTier"',
        f'        name = "{name}"',
        f'        xmin = {_format_number(0)}',
        f'        xmax = {_format_number(xmax)}',
        f'        intervals: size = {len(seg_list)}',
    ]
    for i, seg in enumerate(seg_list, start=1):
        lines += [
            f'        intervals [{i}]:',
            f'            xmin = {_format_number(seg.start_time)}',
            f'            xmax = {_format_number(seg.end_time)}',
            f'            text = "{seg.label}"',
    ]
    return lines


def write_textgrid(
    path: Path,
    *,
    speaker_a: Iterable[Segment],
    speaker_b: Iterable[Segment],
    overlap: Iterable[Segment],
    leak: Iterable[Segment],
    silence_label: str = "#",
) -> None:
    speaker_a_list = list(speaker_a)
    speaker_b_list = list(speaker_b)
    overlap_list = list(overlap)
    leak_list = list(leak)
    all_segments = speaker_a_list + speaker_b_list + overlap_list + leak_list
    xmax = max((seg.end_time for seg in all_segments), default=0.0)

    tiers = []
    tiers += _tier_block(1, "SpeakerA", speaker_a_list, xmax=xmax, silence_label=silence_label)
    tiers += _tier_block(2, "SpeakerB", speaker_b_list, xmax=xmax, silence_label=silence_label)
    tiers += _tier_block(3, "Overlap", overlap_list, xmax=xmax, silence_label=silence_label)
    tiers += _tier_block(4, "Leak", leak_list, xmax=xmax, silence_label=silence_label)
    header = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        f'xmin = {_format_number(0)}',
        f'xmax = {_format_number(xmax)}',
        'tiers? <exists>',
        'size = 4',
        'item []:',
    ]
    lines = header + tiers
    path.write_text("\n".join(lines))


def parse_textgrid(path: Path) -> Dict[str, List[Segment]]:
    content = path.read_text().splitlines()
    tiers: Dict[str, List[Segment]] = {"SpeakerA": [], "SpeakerB": [], "Overlap": [], "Leak": []}
    current = None
    xmin = xmax = 0.0
    for line in content:
        line = line.strip()
        if line.startswith('name ='):
            name = line.split("=", 1)[1].strip().strip('"')
            current = name
        elif line.startswith("xmin"):
            xmin = float(line.split("=")[1])
        elif line.startswith("xmax"):
            xmax = float(line.split("=")[1])
        elif line.startswith("text") and current is not None:
            text = line.split("=", 1)[1].strip().strip('"')
            if current in tiers and text and text != "#":
                tiers[current].append(Segment(start_time=xmin, end_time=xmax, label=text))
    return tiers
