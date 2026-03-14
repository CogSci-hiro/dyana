from __future__ import annotations

from dyana.asr.chunking import (
    MAX_CHUNK_DURATION_SECONDS,
    MIN_CHUNK_DURATION_SECONDS,
    build_asr_chunks,
)
from dyana.decode.ipu import Segment


def test_short_ipus_are_merged_into_longer_chunks() -> None:
    ipus = [
        Segment(start_time=0.0, end_time=0.7, label="A"),
        Segment(start_time=0.8, end_time=1.4, label="A"),
        Segment(start_time=1.5, end_time=2.5, label="A"),
    ]

    chunks = build_asr_chunks(ipus, audio_duration_seconds=3.0)

    assert len(chunks) == 1
    assert chunks[0].ipu_indices == [0, 1, 2]
    assert chunks[0].end_time - chunks[0].start_time >= MIN_CHUNK_DURATION_SECONDS


def test_long_ipus_are_split_before_chunk_assembly() -> None:
    ipus = [Segment(start_time=0.0, end_time=22.0, label="A")]

    chunks = build_asr_chunks(ipus, audio_duration_seconds=22.0)

    assert len(chunks) >= 2
    assert all(chunk.ipu_indices == [0] for chunk in chunks)
    assert all(chunk.end_time - chunk.start_time <= MAX_CHUNK_DURATION_SECONDS for chunk in chunks)


def test_chunks_obey_duration_limits_after_overlap() -> None:
    ipus = [
        Segment(start_time=0.0, end_time=3.0, label="A"),
        Segment(start_time=3.1, end_time=7.0, label="A"),
        Segment(start_time=7.2, end_time=11.0, label="A"),
        Segment(start_time=11.2, end_time=15.5, label="A"),
        Segment(start_time=15.7, end_time=19.5, label="A"),
    ]

    chunks = build_asr_chunks(ipus, audio_duration_seconds=20.0)

    assert chunks
    for chunk in chunks:
        chunk_duration = chunk.end_time - chunk.start_time
        assert chunk_duration >= MIN_CHUNK_DURATION_SECONDS
        assert chunk_duration <= MAX_CHUNK_DURATION_SECONDS
