from __future__ import annotations

from pathlib import Path

from dyana.asr.base import ChunkedASRBackend
from dyana.asr.chunking import ASRChunk
from dyana.asr.transcript import TranscriptSegment, WordTimestamp


class _DummyBackend(ChunkedASRBackend):
    def __init__(self) -> None:
        super().__init__(show_progress=False)
        self.seen_chunks: list[ASRChunk] = []

    @property
    def progress_description(self) -> str:
        return "Dummy progress"

    def merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        return sorted(segments, key=lambda segment: (segment.start_time, segment.end_time, segment.text))

    def transcribe_chunk(self, audio_path: Path, chunk: ASRChunk) -> list[TranscriptSegment]:
        del audio_path
        self.seen_chunks.append(chunk)
        return [
            TranscriptSegment(
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text=f"chunk-{len(self.seen_chunks)}",
                words=[
                    WordTimestamp(
                        word="chunk",
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        confidence=None,
                    )
                ],
            )
        ]


def test_chunked_backend_transcribe_chunks_is_backend_neutral() -> None:
    backend = _DummyBackend()
    chunks = [
        ASRChunk(start_time=1.0, end_time=2.0, ipu_indices=[1]),
        ASRChunk(start_time=0.0, end_time=0.5, ipu_indices=[0]),
    ]

    transcript = backend.transcribe_chunks(Path("dummy.wav"), chunks)

    assert backend.seen_chunks == chunks
    assert [segment.text for segment in transcript.segments] == ["chunk-2", "chunk-1"]
