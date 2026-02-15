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


def count_ipu_starts_after_leak(base_states: Sequence[str]) -> int:
    """
    Count IPU starts whose previous contiguous segment label is LEAK.

    Notes
    -----
    - IPU start labels are A, B, OVL.
    - Uses collapsed contiguous segments from base-state sequence.
    """

    if not base_states:
        return 0

    ipu_labels = {"A", "B", "OVL"}
    segment_labels: List[str] = [base_states[0]]
    for label in base_states[1:]:
        if label != segment_labels[-1]:
            segment_labels.append(label)

    count = 0
    for idx in range(1, len(segment_labels)):
        if segment_labels[idx - 1] == "LEAK" and segment_labels[idx] in ipu_labels:
            count += 1
    return count
