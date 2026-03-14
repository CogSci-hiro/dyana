"""Whisper ASR backend implementation.

Long Whisper decoding windows are more likely to hallucinate trailing words or
invent continuations after long pauses, while very short windows lose the
linguistic context that helps the model stabilize transcripts and timestamps.
The chunking stage therefore keeps windows in a moderate range before this
backend decodes them.

Usage example
-------------
>>> from pathlib import Path
>>> from dyana.asr.chunking import ASRChunk
>>> backend = WhisperBackend(model_name="small")
>>> chunks = [ASRChunk(start_time=0.0, end_time=5.0, ipu_indices=[0, 1])]
>>> isinstance(backend.transcribe_chunks(Path("example.wav"), chunks), Transcript)
True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dyana.asr.chunking import ASRChunk
from dyana.asr.transcript import Transcript, TranscriptSegment, WordTimestamp
from dyana.io.audio import load_audio_mono


DEFAULT_MODEL_NAME = "small"
DEFAULT_DEVICE = "auto"
DEFAULT_TEMPERATURE = 0
DEFAULT_BEAM_SIZE = 5
DUPLICATE_TIME_TOLERANCE_SECONDS = 1e-6


# Whisper backend


class WhisperBackend:
    """Lazy wrapper around the `openai-whisper` transcription library.

    Parameters
    ----------
    model_name
        Whisper model name to load.
    device
        Device string understood by Whisper. ``"auto"`` prefers CUDA when
        available and falls back to CPU.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def transcribe_chunks(self, audio_path: Path, chunks: list[ASRChunk]) -> Transcript:
        """Transcribe the provided chunk list with Whisper.

        Parameters
        ----------
        audio_path
            Path to the input audio file.
        chunks
            Chunk boundaries produced by :func:`dyana.asr.chunking.build_asr_chunks`.

        Returns
        -------
        Transcript
            Transcript containing segment text and word timestamps.
        """

        if not chunks:
            return Transcript(segments=[])

        mono_audio, sample_rate = load_audio_mono(audio_path)
        whisper_model = self._get_model()
        transcript_segments: list[TranscriptSegment] = []

        for chunk in chunks:
            chunk_waveform = _slice_waveform(
                mono_audio,
                sample_rate,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
            if chunk_waveform.size == 0:
                continue

            result = whisper_model.transcribe(
                chunk_waveform,
                temperature=DEFAULT_TEMPERATURE,
                beam_size=DEFAULT_BEAM_SIZE,
                condition_on_previous_text=False,
                word_timestamps=True,
            )

            transcript_segments.extend(
                _convert_whisper_segments(
                    result.get("segments", []),
                    chunk_start_time=chunk.start_time,
                    chunk_end_time=chunk.end_time,
                )
            )

        return Transcript(segments=_merge_adjacent_segments(transcript_segments))

    def _get_model(self) -> Any:
        """Load and cache the Whisper model on first use."""

        if self._model is None:
            whisper = _import_whisper()
            model_device = self.device
            if model_device == DEFAULT_DEVICE:
                model_device = "cuda" if _torch_cuda_is_available() else "cpu"
            self._model = whisper.load_model(self.model_name, device=model_device)
        return self._model


# Conversion helpers


def _import_whisper() -> Any:
    try:
        import whisper
    except ImportError as error:
        raise ImportError(
            "ASR requires the optional 'openai-whisper' package. "
            "Install it before running DYANA with --enable-asr."
        ) from error
    return whisper


def _torch_cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _slice_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """Extract a float32 waveform slice for a chunk."""

    start_sample = max(0, int(round(start_time * sample_rate)))
    end_sample = min(waveform.shape[0], int(round(end_time * sample_rate)))
    if end_sample <= start_sample:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(waveform[start_sample:end_sample], dtype=np.float32)


def _convert_whisper_segments(
    whisper_segments: list[dict[str, Any]],
    *,
    chunk_start_time: float,
    chunk_end_time: float,
) -> list[TranscriptSegment]:
    """Convert Whisper segment payloads into DYANA transcript dataclasses."""

    transcript_segments: list[TranscriptSegment] = []
    for whisper_segment in whisper_segments:
        words = [
            WordTimestamp(
                word=str(word["word"]).strip(),
                start_time=max(chunk_start_time, chunk_start_time + float(word["start"])),
                end_time=min(chunk_end_time, chunk_start_time + float(word["end"])),
                confidence=(
                    None if word.get("probability") is None else float(word["probability"])
                ),
            )
            for word in whisper_segment.get("words", [])
            if str(word.get("word", "")).strip()
        ]

        segment_start_time = chunk_start_time + float(whisper_segment.get("start", 0.0))
        segment_end_time = chunk_start_time + float(
            whisper_segment.get("end", whisper_segment.get("start", 0.0))
        )
        transcript_segments.append(
            TranscriptSegment(
                start_time=max(chunk_start_time, segment_start_time),
                end_time=min(chunk_end_time, segment_end_time),
                text=str(whisper_segment.get("text", "")).strip(),
                words=words,
            )
        )

    return transcript_segments


def _merge_adjacent_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Collapse exact duplicates that can appear because chunk overlap repeats context."""

    merged_segments: list[TranscriptSegment] = []
    for segment in sorted(segments, key=lambda item: (item.start_time, item.end_time, item.text)):
        if not segment.text and not segment.words:
            continue
        if merged_segments:
            previous_segment = merged_segments[-1]
            if (
                abs(previous_segment.start_time - segment.start_time) < DUPLICATE_TIME_TOLERANCE_SECONDS
                and abs(previous_segment.end_time - segment.end_time) < DUPLICATE_TIME_TOLERANCE_SECONDS
                and previous_segment.text == segment.text
            ):
                continue
        merged_segments.append(segment)
    return merged_segments
