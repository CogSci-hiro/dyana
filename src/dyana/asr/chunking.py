"""Chunk IPUs into ASR-friendly windows.

Whisper-style sequence-to-sequence ASR models can drift or hallucinate when a
single decoding window becomes too long, especially across pauses or topic
changes. Very short windows are also undesirable because they provide too
little acoustic and linguistic context, which can hurt word recognition and
timestamp quality.

This module mitigates both issues by:

1. ignoring extremely short IPUs that are likely unstable for ASR,
2. merging short neighboring IPUs into longer windows,
3. splitting very long IPUs before they become hallucination-prone, and
4. adding a small overlap so chunk boundaries do not clip words.

Usage example
-------------
>>> from dyana.decode.ipu import Segment
>>> ipus = [
...     Segment(start_time=0.0, end_time=0.8, label="A"),
...     Segment(start_time=1.0, end_time=2.4, label="A"),
... ]
>>> build_asr_chunks(ipus, audio_duration_seconds=3.0)
[ASRChunk(start_time=0.0, end_time=2.7, ipu_indices=[0, 1])]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


MIN_CHUNK_DURATION_SECONDS = 2.0
TARGET_CHUNK_DURATION_SECONDS = 10.0
MAX_CHUNK_DURATION_SECONDS = 18.0
MAX_IPU_DURATION_SECONDS = 15.0
CHUNK_OVERLAP_SECONDS = 0.3
MIN_IPU_DURATION_SECONDS = 0.3


# Public chunk dataclass


@dataclass(frozen=True)
class ASRChunk:
    """ASR-ready chunk derived from one or more IPUs.

    Parameters
    ----------
    start_time
        Chunk start time in seconds.
    end_time
        Chunk end time in seconds.
    ipu_indices
        Indices of source IPUs contributing acoustic content to the chunk.
    """

    start_time: float
    end_time: float
    ipu_indices: list[int]


# Internal helpers


class _IPULike(Protocol):
    start_time: float
    end_time: float


@dataclass(frozen=True)
class _ChunkUnit:
    """Intermediate unit used to assemble final chunks."""

    start_time: float
    end_time: float
    ipu_indices: list[int]


# Chunking algorithm


def build_asr_chunks(ipus: Sequence[_IPULike], audio_duration_seconds: float) -> list[ASRChunk]:
    """Build ASR-friendly chunks from IPUs.

    Parameters
    ----------
    ipus
        Ordered inter-pausal units. Each item must expose ``start_time`` and
        ``end_time`` in seconds.
    audio_duration_seconds
        Total audio duration used to clamp overlap-expanded boundaries.

    Returns
    -------
    list[ASRChunk]
        Ordered ASR chunks with modest boundary overlap.
    """

    if audio_duration_seconds < 0.0:
        raise ValueError("audio_duration_seconds must be non-negative.")
    if not ipus:
        return []

    filtered_units = _filter_and_split_ipus(ipus)
    merged_units = _merge_short_units(filtered_units)
    return _assemble_chunks(merged_units, audio_duration_seconds)


def _filter_and_split_ipus(ipus: Sequence[_IPULike]) -> list[_ChunkUnit]:
    """Drop unstable micro-IPUs and split overly long IPUs."""

    chunk_units: list[_ChunkUnit] = []
    for ipu_index, ipu in enumerate(ipus):
        ipu_start = float(ipu.start_time)
        ipu_end = float(ipu.end_time)
        ipu_duration = ipu_end - ipu_start

        if ipu_duration < MIN_IPU_DURATION_SECONDS:
            continue

        if ipu_duration <= MAX_IPU_DURATION_SECONDS:
            chunk_units.append(
                _ChunkUnit(start_time=ipu_start, end_time=ipu_end, ipu_indices=[ipu_index])
            )
            continue

        # Split very long IPUs into roughly target-sized spans to reduce
        # long-context drift and hallucinated continuations during decoding.
        split_start = ipu_start
        while split_start < ipu_end:
            split_end = min(split_start + TARGET_CHUNK_DURATION_SECONDS, ipu_end)
            chunk_units.append(
                _ChunkUnit(start_time=split_start, end_time=split_end, ipu_indices=[ipu_index])
            )
            split_start = split_end

    return chunk_units


def _merge_short_units(chunk_units: Sequence[_ChunkUnit]) -> list[_ChunkUnit]:
    """Merge neighboring short units until they become ASR-usable."""

    if not chunk_units:
        return []

    merged_units: list[_ChunkUnit] = []
    current_start = chunk_units[0].start_time
    current_end = chunk_units[0].end_time
    current_indices = list(chunk_units[0].ipu_indices)

    for chunk_unit in chunk_units[1:]:
        current_duration = current_end - current_start
        proposed_end = chunk_unit.end_time
        proposed_duration = proposed_end - current_start

        if current_duration < MIN_CHUNK_DURATION_SECONDS:
            current_end = proposed_end
            current_indices.extend(chunk_unit.ipu_indices)
            continue

        if proposed_duration <= TARGET_CHUNK_DURATION_SECONDS:
            current_end = proposed_end
            current_indices.extend(chunk_unit.ipu_indices)
            continue

        merged_units.append(
            _ChunkUnit(start_time=current_start, end_time=current_end, ipu_indices=current_indices)
        )
        current_start = chunk_unit.start_time
        current_end = chunk_unit.end_time
        current_indices = list(chunk_unit.ipu_indices)

    merged_units.append(
        _ChunkUnit(start_time=current_start, end_time=current_end, ipu_indices=current_indices)
    )
    return merged_units


def _assemble_chunks(
    chunk_units: Sequence[_ChunkUnit], audio_duration_seconds: float
) -> list[ASRChunk]:
    """Assemble final chunks near the target duration while respecting limits."""

    assembled_chunks: list[ASRChunk] = []
    unit_index = 0

    while unit_index < len(chunk_units):
        chunk_start = chunk_units[unit_index].start_time
        chunk_end = chunk_units[unit_index].end_time
        chunk_ipu_indices = list(chunk_units[unit_index].ipu_indices)
        unit_index += 1

        # Keep adding IPUs until we reach the target window length, but stop
        # before growing beyond the maximum hallucination-safe duration.
        while unit_index < len(chunk_units):
            next_unit = chunk_units[unit_index]
            candidate_end = next_unit.end_time
            candidate_duration = candidate_end - chunk_start

            if candidate_duration > MAX_CHUNK_DURATION_SECONDS:
                break

            chunk_end = candidate_end
            chunk_ipu_indices.extend(next_unit.ipu_indices)
            unit_index += 1

            if chunk_end - chunk_start >= TARGET_CHUNK_DURATION_SECONDS:
                break

        core_duration = chunk_end - chunk_start
        overlap_budget = max(0.0, MAX_CHUNK_DURATION_SECONDS - core_duration)
        left_overlap = min(
            CHUNK_OVERLAP_SECONDS,
            chunk_start,
            overlap_budget / 2.0,
        )
        right_overlap = min(
            CHUNK_OVERLAP_SECONDS,
            max(0.0, audio_duration_seconds - chunk_end),
            overlap_budget - left_overlap,
        )
        overlap_start = chunk_start - left_overlap
        overlap_end = chunk_end + right_overlap
        assembled_chunks.append(
            ASRChunk(
                start_time=overlap_start,
                end_time=overlap_end,
                ipu_indices=sorted(set(chunk_ipu_indices)),
            )
        )

    return assembled_chunks
