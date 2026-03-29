from __future__ import annotations

from pathlib import Path

from dyana.asr.transcript import (
    Transcript,
    TranscriptSegment,
    WordTimestamp,
    align_transcript_to_ipus,
    assign_speaker,
    merge_transcripts,
    write_textgrid,
)
from dyana.decode.ipu import Segment


def test_write_textgrid_writes_segment_and_word_tiers(tmp_path: Path) -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                start_time=0.0,
                end_time=1.0,
                text="bonjour monde",
                words=[
                    WordTimestamp(word="bonjour", start_time=0.0, end_time=0.4, confidence=0.9),
                    WordTimestamp(word="monde", start_time=0.5, end_time=1.0, confidence=0.8),
                ],
            )
        ]
    )
    path = tmp_path / "transcript.TextGrid"

    write_textgrid(transcript, path)

    content = path.read_text()
    assert 'name = "segments"' in content
    assert 'name = "words"' in content
    assert 'text = "bonjour monde"' in content
    assert 'text = "bonjour"' in content


def test_write_textgrid_writes_per_speaker_tiers(tmp_path: Path) -> None:
    transcript_a = assign_speaker(
        Transcript(
            segments=[
                TranscriptSegment(
                    start_time=0.0,
                    end_time=1.0,
                    text="bonjour",
                    words=[WordTimestamp(word="bonjour", start_time=0.0, end_time=1.0, confidence=0.9)],
                )
            ]
        ),
        "A",
    )
    transcript_b = assign_speaker(
        Transcript(
            segments=[
                TranscriptSegment(
                    start_time=1.5,
                    end_time=2.0,
                    text="salut",
                    words=[WordTimestamp(word="salut", start_time=1.5, end_time=2.0, confidence=0.8)],
                )
            ]
        ),
        "B",
    )
    path = tmp_path / "speaker_transcript.TextGrid"

    write_textgrid(merge_transcripts([transcript_a, transcript_b]), path)

    content = path.read_text()
    assert 'name = "segments_A"' in content
    assert 'name = "words_A"' in content
    assert 'name = "segments_B"' in content
    assert 'name = "words_B"' in content


def test_merge_transcripts_orders_by_time() -> None:
    transcript_a = assign_speaker(
        Transcript(
            segments=[
                TranscriptSegment(
                    start_time=2.0,
                    end_time=3.0,
                    text="late",
                    words=[],
                )
            ]
        ),
        "A",
    )
    transcript_b = assign_speaker(
        Transcript(
            segments=[
                TranscriptSegment(
                    start_time=1.0,
                    end_time=1.5,
                    text="early",
                    words=[],
                )
            ]
        ),
        "B",
    )

    transcript = merge_transcripts([transcript_a, transcript_b])

    assert [segment.text for segment in transcript.segments] == ["early", "late"]
    assert [segment.speaker for segment in transcript.segments] == ["B", "A"]


def test_align_transcript_to_ipus_snaps_words_to_ipu_boundaries() -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                start_time=0.0,
                end_time=2.0,
                text="bonjour salut",
                words=[
                    WordTimestamp(word="bonjour", start_time=0.1, end_time=0.8, confidence=0.9),
                    WordTimestamp(word="salut", start_time=1.2, end_time=1.8, confidence=0.8),
                ],
            )
        ]
    )
    ipus = [
        Segment(start_time=0.0, end_time=1.0, label="A"),
        Segment(start_time=1.0, end_time=2.0, label="A"),
    ]

    aligned = align_transcript_to_ipus(transcript, ipus, speaker="A")

    assert [segment.start_time for segment in aligned.segments] == [0.0, 1.0]
    assert [segment.end_time for segment in aligned.segments] == [1.0, 2.0]
    assert [segment.text for segment in aligned.segments] == ["bonjour", "salut"]
    assert all(segment.speaker == "A" for segment in aligned.segments)


def test_align_transcript_to_ipus_deduplicates_overlap_words() -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                start_time=0.0,
                end_time=1.0,
                text="bonjour",
                words=[WordTimestamp(word="bonjour", start_time=0.2, end_time=0.7, confidence=0.9)],
            ),
            TranscriptSegment(
                start_time=0.0,
                end_time=1.0,
                text="bonjour",
                words=[WordTimestamp(word="bonjour", start_time=0.2004, end_time=0.7002, confidence=0.8)],
            ),
        ]
    )
    ipus = [Segment(start_time=0.0, end_time=1.0, label="A")]

    aligned = align_transcript_to_ipus(transcript, ipus, speaker="A")

    assert len(aligned.segments) == 1
    assert len(aligned.segments[0].words) == 1
