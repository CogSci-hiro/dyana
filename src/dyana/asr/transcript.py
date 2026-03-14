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

from dataclasses import asdict, dataclass
from typing import Any, Mapping


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
    """

    word: str
    start_time: float
    end_time: float
    confidence: float | None


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
    """

    start_time: float
    end_time: float
    text: str
    words: list[WordTimestamp]


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
                    )
                    for word in segment.get("words", [])
                ],
            )
            for segment in segments_payload
        ]
        return cls(segments=segments)
