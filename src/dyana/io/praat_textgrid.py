"""Praat TextGrid I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

from dyana.decode.ipu import Segment


def _tier_block(name: str, segments: Iterable[Segment]) -> List[str]:
    seg_list = list(segments)
    lines = [
        f'    item []: {1}',
        '        class = "IntervalTier"',
        f'        name = "{name}"',
        f'        xmin = {0}',
        f'        xmax = {seg_list[-1].end_time if seg_list else 0}',
        f'        intervals: size = {len(seg_list)}',
    ]
    for i, seg in enumerate(seg_list, start=1):
        lines += [
            f'        intervals [{i}]:',
            f'            xmin = {seg.start_time}',
            f'            xmax = {seg.end_time}',
            f'            text = "{seg.label}"',
        ]
    return lines


def write_textgrid(path: Path, *, speaker_a: Iterable[Segment], speaker_b: Iterable[Segment], overlap: Iterable[Segment], leak: Iterable[Segment]) -> None:
    tiers = []
    tiers += _tier_block("SpeakerA", speaker_a)
    tiers += _tier_block("SpeakerB", speaker_b)
    tiers += _tier_block("Overlap", overlap)
    tiers += _tier_block("Leak", leak)
    header = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        '',
        'xmin = 0',
        f'xmax = {tiers[-2] if tiers else 0}',
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
            if current in tiers and text:
                tiers[current].append(Segment(start_time=xmin, end_time=xmax, label=text))
    return tiers
