"""IPU construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from dyana.core.timebase import TimeBase


@dataclass
class Segment:
    start_time: float
    end_time: float
    label: str


def extract_ipus(states: Sequence[str], timebase: TimeBase, target_label: str, *, min_duration_s: float) -> List[Segment]:
    segments: List[Segment] = []
    start = None
    for idx, state in enumerate(states):
        if state == target_label:
            if start is None:
                start = idx
        else:
            if start is not None:
                end = idx
                dur = (end - start) * timebase.hop_s
                if dur >= min_duration_s:
                    segments.append(
                        Segment(
                            start_time=timebase.frame_to_time(start),
                            end_time=timebase.frame_to_time(end),
                            label=target_label,
                        )
                    )
                start = None
    if start is not None:
        end = len(states)
        dur = (end - start) * timebase.hop_s
        if dur >= min_duration_s:
            segments.append(
                Segment(
                    start_time=timebase.frame_to_time(start),
                    end_time=timebase.frame_to_time(end),
                    label=target_label,
                )
            )
    return segments
