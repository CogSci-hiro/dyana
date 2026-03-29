"""Backend-neutral interfaces for chunked ASR.

This module centralizes chunk-level iteration so progress reporting is shared
across ASR backends instead of being implemented separately in each backend.
Backends only need to decode a single :class:`~dyana.asr.chunking.ASRChunk`
and return transcript segments for that chunk.
"""

from __future__ import annotations

from contextlib import contextmanager
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterator

from dyana.asr.chunking import ASRChunk
from dyana.asr.transcript import Transcript, TranscriptSegment


ENABLE_PROGRESS_DEFAULT = True


# Shared chunked backend base class


class ChunkedASRBackend(ABC):
    """Abstract base class for chunk-oriented ASR backends.

    Parameters
    ----------
    show_progress
        Whether chunk-level progress should be shown during transcription.
    """

    def __init__(self, *, show_progress: bool = ENABLE_PROGRESS_DEFAULT) -> None:
        self.show_progress = show_progress

    def transcribe_chunks(self, audio_path: Path, chunks: list[ASRChunk]) -> Transcript:
        """Transcribe a list of chunks and merge the resulting segments.

        Parameters
        ----------
        audio_path
            Path to the source audio file.
        chunks
            Ordered list of ASR chunks to decode.

        Returns
        -------
        Transcript
            Transcript built from all decoded chunks.
        """

        if not chunks:
            return Transcript(segments=[])

        transcript_segments: list[TranscriptSegment] = []
        with chunk_progress(
            chunks,
            description=self.progress_description,
            enabled=self.show_progress,
        ) as advance_progress:
            for chunk in chunks:
                transcript_segments.extend(self.transcribe_chunk(audio_path, chunk))
                advance_progress(chunk)
        return Transcript(segments=self.merge_segments(transcript_segments))

    @property
    def progress_description(self) -> str:
        """Human-readable label for chunk transcription progress."""

        return "Transcribing"

    def merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Merge or deduplicate transcript segments after chunk decoding."""

        return segments

    @abstractmethod
    def transcribe_chunk(self, audio_path: Path, chunk: ASRChunk) -> list[TranscriptSegment]:
        """Transcribe a single chunk into one or more transcript segments."""


# Shared progress helper


@contextmanager
def chunk_progress(
    chunks: list[ASRChunk],
    *,
    description: str,
    enabled: bool,
) -> Iterator[Callable[[ASRChunk], None]]:
    """Create a backend-neutral ASR progress reporter with ETA.

    Parameters
    ----------
    chunks
        Chunks to iterate over.
    description
        Progress bar label shown to the user.
    enabled
        Whether progress reporting is enabled.

    Returns
    -------
    Iterator[Callable[[ASRChunk], None]]
        Context manager yielding a callback that should be called after each
        completed chunk.
    """

    if not enabled:
        yield lambda _chunk: None
        return

    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        yield lambda _chunk: None
        return

    total_audio_seconds = sum(max(0.0, chunk.end_time - chunk.start_time) for chunk in chunks)
    total_chunks = len(chunks)

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.fields[completed_chunks]}/{task.fields[total_chunks]} chunks)"),
        TextColumn("{task.completed:.1f}/{task.total:.1f}s"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task_id = progress.add_task(
            description,
            total=total_audio_seconds,
            completed_chunks=0,
            total_chunks=total_chunks,
        )

        def _advance(chunk: ASRChunk) -> None:
            chunk_duration = max(0.0, chunk.end_time - chunk.start_time)
            task = progress.tasks[task_id]
            completed_chunks = int(task.fields["completed_chunks"]) + 1
            progress.update(
                task_id,
                advance=chunk_duration,
                completed_chunks=completed_chunks,
                total_chunks=total_chunks,
            )

        yield _advance
