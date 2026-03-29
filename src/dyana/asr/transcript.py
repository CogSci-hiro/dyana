"""Transcript structures for automatic speech recognition outputs.

The dataclasses in this module store segment-level text and word-level timing
information produced by an ASR backend. Word timestamps are preserved because
downstream conversational analysis often needs precise lexical timing rather
than segment-only text.

Usage example
-------------
>>> transcript = Transcript(
...     segments=[
...         TranscriptSegment(
...             start_time=0.0,
...             end_time=1.2,
...             text="hello there",
...             words=[
...                 WordTimestamp("hello", 0.0, 0.5, 0.92),
...                 WordTimestamp("there", 0.6, 1.2, 0.88),
...             ],
...         )
...     ]
... )
>>> payload = transcript.to_json()
>>> Transcript.from_json(payload) == transcript
True
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol


WORD_JOINER = " "
WORD_TIME_TOLERANCE_SECONDS = 1e-3


# Transcript dataclasses


@dataclass(frozen=True)
class WordTimestamp:
    """Word-level timestamp emitted by an ASR backend.

    Parameters
    ----------
    word
        Recognized word text.
    start_time
        Word start time in seconds.
    end_time
        Word end time in seconds.
    confidence
        Optional backend confidence score for the word.
    speaker
        Optional speaker label for the word.
    """

    word: str
    start_time: float
    end_time: float
    confidence: float | None
    speaker: str | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    """Continuous transcript span with optional word timestamps.

    Parameters
    ----------
    start_time
        Segment start time in seconds.
    end_time
        Segment end time in seconds.
    text
        Recognized text for the segment.
    words
        Word-level timing entries aligned to the segment.
    speaker
        Optional speaker label for the segment.
    """

    start_time: float
    end_time: float
    text: str
    words: list[WordTimestamp]
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    """Transcript returned by an ASR backend.

    Parameters
    ----------
    segments
        Ordered transcript segments.
    """

    segments: list[TranscriptSegment]

    def to_json(self) -> dict[str, Any]:
        """Serialize the transcript into a JSON-compatible payload.

        Returns
        -------
        dict[str, Any]
            Human-readable JSON payload that can be passed to ``json.dumps``.
        """

        return {
            "type": "asr_transcript",
            "segments": [asdict(segment) for segment in self.segments],
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Transcript":
        """Deserialize a transcript from a JSON-compatible mapping.

        Parameters
        ----------
        payload
            Mapping produced by :meth:`to_json`.

        Returns
        -------
        Transcript
            Reconstructed transcript object.
        """

        segments_payload = payload.get("segments", [])
        segments = [
            TranscriptSegment(
                start_time=float(segment["start_time"]),
                end_time=float(segment["end_time"]),
                text=str(segment["text"]),
                words=[
                    WordTimestamp(
                        word=str(word["word"]),
                        start_time=float(word["start_time"]),
                        end_time=float(word["end_time"]),
                        confidence=(
                            None if word.get("confidence") is None else float(word["confidence"])
                        ),
                        speaker=None if word.get("speaker") is None else str(word["speaker"]),
                    )
                    for word in segment.get("words", [])
                ],
                speaker=None if segment.get("speaker") is None else str(segment["speaker"]),
            )
            for segment in segments_payload
        ]
        return cls(segments=segments)


def assign_speaker(transcript: Transcript, speaker: str) -> Transcript:
    """Return a copy of the transcript with a speaker label attached.

    Parameters
    ----------
    transcript
        Transcript to annotate.
    speaker
        Speaker label to assign to all segments and words.

    Returns
    -------
    Transcript
        Speaker-annotated transcript.
    """

    return Transcript(
        segments=[
            replace(
                segment,
                speaker=speaker,
                words=[replace(word, speaker=speaker) for word in segment.words],
            )
            for segment in transcript.segments
        ]
    )


def merge_transcripts(transcripts: list[Transcript]) -> Transcript:
    """Merge multiple transcripts into a single time-ordered transcript.

    Parameters
    ----------
    transcripts
        Transcript objects to merge.

    Returns
    -------
    Transcript
        Combined transcript sorted by timing.
    """

    merged_segments = [
        segment
        for transcript in transcripts
        for segment in transcript.segments
    ]
    merged_segments.sort(key=lambda item: (item.start_time, item.end_time, item.speaker or "", item.text))
    return Transcript(segments=merged_segments)


class _IPULike(Protocol):
    """Protocol for IPU-like segments used in transcript alignment."""

    start_time: float
    end_time: float
    label: str


def align_transcript_to_ipus(
    transcript: Transcript,
    ipus: list[_IPULike],
    *,
    speaker: str | None = None,
) -> Transcript:
    """Project transcript words back onto IPU-aligned transcript segments.

    Parameters
    ----------
    transcript
        Chunk-level transcript emitted by the ASR backend.
    ipus
        Source IPUs used to build the ASR chunks.
    speaker
        Optional speaker label to stamp onto the aligned segments and words.

    Returns
    -------
    Transcript
        Transcript whose segment boundaries match the contributing IPUs.
    """

    if not ipus:
        return Transcript(segments=[])

    unique_words = _deduplicate_words(transcript)
    aligned_segments: list[TranscriptSegment] = []
    for ipu in ipus:
        ipu_words = [
            _clip_word_to_interval(word, start_time=ipu.start_time, end_time=ipu.end_time, speaker=speaker or ipu.label)
            for word in unique_words
            if _word_belongs_to_ipu(word, ipu)
        ]
        ipu_words = [word for word in ipu_words if word is not None]
        if not ipu_words:
            continue

        aligned_segments.append(
            TranscriptSegment(
                start_time=ipu.start_time,
                end_time=ipu.end_time,
                text=WORD_JOINER.join(word.word for word in ipu_words),
                words=ipu_words,
                speaker=speaker or ipu.label,
            )
        )
    return Transcript(segments=aligned_segments)


def write_textgrid(transcript: Transcript, path: Path) -> None:
    """Write a transcript TextGrid with segment and word tiers.

    Parameters
    ----------
    transcript
        Transcript to export.
    path
        Destination TextGrid path.
    """

    interval_payload = _build_textgrid_payload(transcript)
    xmax = interval_payload["xmax"]
    tiers = interval_payload["tiers"]
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
    lines = header + [line for tier in tiers for line in tier]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _deduplicate_words(transcript: Transcript) -> list[WordTimestamp]:
    """Remove overlap-induced duplicate words while preserving time order."""

    deduplicated_words: list[WordTimestamp] = []
    seen_keys: set[tuple[str, int, int, str | None]] = set()
    all_words = [
        word
        for segment in transcript.segments
        for word in segment.words
    ]
    all_words.sort(key=lambda item: (item.start_time, item.end_time, item.word, item.speaker or ""))
    for word in all_words:
        word_key = (
            word.word,
            int(round(word.start_time / WORD_TIME_TOLERANCE_SECONDS)),
            int(round(word.end_time / WORD_TIME_TOLERANCE_SECONDS)),
            word.speaker,
        )
        if word_key in seen_keys:
            continue
        seen_keys.add(word_key)
        deduplicated_words.append(word)
    return deduplicated_words


def _word_belongs_to_ipu(word: WordTimestamp, ipu: _IPULike) -> bool:
    """Decide whether a word should be assigned to an IPU."""

    midpoint = (word.start_time + word.end_time) / 2.0
    if ipu.start_time <= midpoint <= ipu.end_time:
        return True
    overlap_start = max(word.start_time, ipu.start_time)
    overlap_end = min(word.end_time, ipu.end_time)
    return overlap_end > overlap_start


def _clip_word_to_interval(
    word: WordTimestamp,
    *,
    start_time: float,
    end_time: float,
    speaker: str | None,
) -> WordTimestamp | None:
    """Clip a word to an enclosing interval."""

    clipped_start = max(word.start_time, start_time)
    clipped_end = min(word.end_time, end_time)
    if clipped_end <= clipped_start:
        return None
    return replace(
        word,
        start_time=clipped_start,
        end_time=clipped_end,
        speaker=speaker,
    )


def _build_textgrid_payload(transcript: Transcript) -> dict[str, Any]:
    """Build TextGrid tiers for transcript segments and words."""

    speaker_labels = sorted(
        {
            segment.speaker
            for segment in transcript.segments
            if segment.speaker is not None
        }
    )
    xmax = max(
        [0.0]
        + [segment.end_time for segment in transcript.segments if segment.end_time > segment.start_time]
        + [
            word.end_time
            for segment in transcript.segments
            for word in segment.words
            if word.end_time > word.start_time
        ]
    )

    tiers: list[list[str]]
    if speaker_labels:
        tiers = _build_speaker_tiers(transcript, speaker_labels, xmax=xmax)
    else:
        segment_intervals = [
            (segment.start_time, segment.end_time, segment.text)
            for segment in transcript.segments
            if segment.end_time > segment.start_time
        ]
        word_intervals = [
            (word.start_time, word.end_time, word.word)
            for segment in transcript.segments
            for word in segment.words
            if word.end_time > word.start_time
        ]
        tiers = [
            _tier_block(1, "segments", segment_intervals, xmax=xmax),
            _tier_block(2, "words", word_intervals, xmax=xmax),
        ]
    return {"xmax": xmax, "tiers": tiers}


def _build_speaker_tiers(
    transcript: Transcript,
    speaker_labels: list[str],
    *,
    xmax: float,
) -> list[list[str]]:
    """Build one segment tier and one word tier per speaker."""

    tiers: list[list[str]] = []
    tier_index = 1
    for speaker in speaker_labels:
        speaker_segments = [
            (segment.start_time, segment.end_time, segment.text)
            for segment in transcript.segments
            if segment.speaker == speaker and segment.end_time > segment.start_time
        ]
        speaker_words = [
            (word.start_time, word.end_time, word.word)
            for segment in transcript.segments
            if segment.speaker == speaker
            for word in segment.words
            if word.end_time > word.start_time
        ]
        tiers.append(_tier_block(tier_index, f"segments_{speaker}", speaker_segments, xmax=xmax))
        tier_index += 1
        tiers.append(_tier_block(tier_index, f"words_{speaker}", speaker_words, xmax=xmax))
        tier_index += 1
    return tiers


def _tier_block(index: int, name: str, intervals: list[tuple[float, float, str]], *, xmax: float) -> list[str]:
    """Create a single Praat interval tier."""

    filled_intervals = _fill_gaps(intervals, xmax=xmax)
    lines = [
        f"    item [{index}]:",
        '        class = "IntervalTier"',
        f'        name = "{name}"',
        f"        xmin = {_format_number(0.0)}",
        f"        xmax = {_format_number(xmax)}",
        f"        intervals: size = {len(filled_intervals)}",
    ]
    for interval_index, (start, end, label) in enumerate(filled_intervals, start=1):
        lines.extend(
            [
                f"        intervals [{interval_index}]:",
                f"            xmin = {_format_number(start)}",
                f"            xmax = {_format_number(end)}",
                f'            text = "{label}"',
            ]
        )
    return lines


def _fill_gaps(intervals: list[tuple[float, float, str]], *, xmax: float) -> list[tuple[float, float, str]]:
    """Fill gaps in interval tiers with empty labels."""

    if not intervals:
        return [(0.0, xmax, "")]

    ordered_intervals = sorted(
        [
            (start, end, label)
            for start, end, label in intervals
            if end > start
        ],
        key=lambda item: (item[0], item[1], item[2]),
    )
    if not ordered_intervals:
        return [(0.0, xmax, "")]

    filled_intervals: list[tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, label in ordered_intervals:
        if start > cursor:
            filled_intervals.append((cursor, start, ""))
        normalized_start = max(cursor, start)
        if end <= normalized_start:
            continue
        filled_intervals.append((normalized_start, end, label))
        cursor = end
    if cursor < xmax:
        filled_intervals.append((cursor, xmax, ""))
    return filled_intervals


def _format_number(value: float) -> str:
    """Format float values in Praat-friendly style."""

    return f"{float(value):.6f}".rstrip("0").rstrip(".")
