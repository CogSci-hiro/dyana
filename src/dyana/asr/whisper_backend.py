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

import os
import ssl
from pathlib import Path
from typing import Any
from urllib.error import URLError

import numpy as np

from dyana.asr.base import ChunkedASRBackend
from dyana.asr.chunking import ASRChunk
from dyana.asr.transcript import Transcript, TranscriptSegment, WordTimestamp
from dyana.io.audio import load_audio_mono


DEFAULT_MODEL_NAME = "small"
DEFAULT_DEVICE = "auto"
DEFAULT_TEMPERATURE = 0
DEFAULT_BEAM_SIZE = 5
WHISPER_SAMPLE_RATE = 16000
DUPLICATE_TIME_TOLERANCE_SECONDS = 1e-6
MIN_INTERVAL_DURATION_SECONDS = 1e-4
DEFAULT_WHISPER_CACHE_DIR = Path.home() / ".cache" / "whisper"
ENV_WHISPER_MODEL_DIR = "WHISPER_MODEL_DIR"
ENV_WHISPER_MODEL_PATH = "WHISPER_MODEL_PATH"
ENV_WHISPER_LANGUAGE = "WHISPER_LANGUAGE"


# Backend-specific errors


class WhisperModelLoadError(RuntimeError):
    """Raised when the Whisper model cannot be loaded predictably."""


# Whisper backend


class WhisperBackend(ChunkedASRBackend):
    """Lazy wrapper around the `openai-whisper` transcription library.

    Parameters
    ----------
    model_name
        Whisper model name to load.
    device
        Device string understood by Whisper. ``"auto"`` prefers CUDA when
        available and falls back to CPU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        model_path: Path | None = None,
        model_dir: Path | None = None,
        language: str | None = None,
        audio_channel: int | None = None,
        show_progress: bool = True,
    ) -> None:
        super().__init__(show_progress=show_progress)
        self.model_name = model_name
        self.device = device
        self.model_path = Path(model_path) if model_path is not None else _env_path(ENV_WHISPER_MODEL_PATH)
        self.model_dir = Path(model_dir) if model_dir is not None else _env_path(ENV_WHISPER_MODEL_DIR)
        self.language = language or os.environ.get(ENV_WHISPER_LANGUAGE)
        self.audio_channel = audio_channel
        self._model: Any | None = None

    @property
    def progress_description(self) -> str:
        """Human-readable progress label for Whisper chunk decoding."""

        return f"Transcribing ({self.model_name})"

    def merge_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Deduplicate overlap-repeated Whisper transcript segments."""

        return _merge_adjacent_segments(segments)

    def transcribe_chunk(self, audio_path: Path, chunk: ASRChunk) -> list[TranscriptSegment]:
        """Transcribe a single chunk with Whisper.

        Parameters
        ----------
        audio_path
            Path to the input audio file.
        chunk
            Chunk boundary produced by :func:`dyana.asr.chunking.build_asr_chunks`.

        Returns
        -------
        list[TranscriptSegment]
            Transcript segments decoded for the chunk.
        """

        mono_audio, sample_rate = load_audio_mono(audio_path, channel=self.audio_channel)
        whisper_model = self._get_model()
        chunk_waveform = _slice_waveform(
            mono_audio,
            sample_rate,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
        )
        if chunk_waveform.size == 0:
            return []
        if sample_rate != WHISPER_SAMPLE_RATE:
            chunk_waveform = _resample_linear(
                chunk_waveform,
                sample_rate,
                WHISPER_SAMPLE_RATE,
            )

        result = whisper_model.transcribe(
            chunk_waveform,
            temperature=DEFAULT_TEMPERATURE,
            beam_size=DEFAULT_BEAM_SIZE,
            condition_on_previous_text=False,
            word_timestamps=True,
            language=self.language,
        )

        return _convert_whisper_segments(
            result.get("segments", []),
            chunk_start_time=chunk.start_time,
            chunk_end_time=chunk.end_time,
        )

    def _get_model(self) -> Any:
        """Load and cache the Whisper model on first use."""

        if self._model is None:
            whisper = _import_whisper()
            model_device = self.device
            if model_device == DEFAULT_DEVICE:
                model_device = "cuda" if _torch_cuda_is_available() else "cpu"
            model_source = self._resolve_model_source()
            try:
                self._model = whisper.load_model(
                    str(model_source),
                    device=model_device,
                    download_root=str(self.get_model_cache_dir()),
                )
            except FileNotFoundError as error:
                raise WhisperModelLoadError(
                    f"Whisper model file not found: {model_source}\n"
                    "Pass --asr-model-path to a local checkpoint file or place the model in "
                    f"{self.get_model_cache_dir()}."
                ) from error
            except URLError as error:
                raise _build_download_error(
                    error=error,
                    model_name=self.model_name,
                    model_cache_dir=self.get_model_cache_dir(),
                    model_path=self.model_path,
                ) from error
            except ssl.SSLCertVerificationError as error:
                raise _build_ssl_error(
                    error=error,
                    model_name=self.model_name,
                    model_cache_dir=self.get_model_cache_dir(),
                    model_path=self.model_path,
                ) from error
        return self._model

    def get_model_cache_dir(self) -> Path:
        """Return the effective Whisper cache directory."""

        return self.model_dir or DEFAULT_WHISPER_CACHE_DIR

    def get_expected_model_path(self) -> Path:
        """Return the expected checkpoint path for the selected model."""

        return self.get_model_cache_dir() / f"{self.model_name}.pt"

    def _resolve_model_source(self) -> Path | str:
        """Resolve either a direct checkpoint path or a model name."""

        if self.model_path is not None:
            return self.model_path
        return self.model_name


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


def _env_path(variable_name: str) -> Path | None:
    raw_value = os.environ.get(variable_name)
    if raw_value is None or raw_value.strip() == "":
        return None
    return Path(raw_value).expanduser()


def _build_download_error(
    *,
    error: URLError,
    model_name: str,
    model_cache_dir: Path,
    model_path: Path | None,
) -> WhisperModelLoadError:
    """Build a user-facing error for Whisper download failures."""

    if isinstance(error.reason, ssl.SSLCertVerificationError):
        return _build_ssl_error(
            error=error.reason,
            model_name=model_name,
            model_cache_dir=model_cache_dir,
            model_path=model_path,
        )

    expected_path = model_path or model_cache_dir / f"{model_name}.pt"
    return WhisperModelLoadError(
        "DYANA could not download the Whisper model automatically.\n"
        f"Model: {model_name}\n"
        f"Expected local checkpoint path: {expected_path}\n"
        f"Whisper cache directory: {model_cache_dir}\n"
        "You can fix this by either:\n"
        "1. passing --asr-model-path /path/to/model.pt\n"
        "2. setting WHISPER_MODEL_DIR to a cache directory containing the checkpoint\n"
        "3. running `dyana asr-setup --model "
        f"{model_name}` to see the expected local path"
    )


def _build_ssl_error(
    *,
    error: ssl.SSLCertVerificationError,
    model_name: str,
    model_cache_dir: Path,
    model_path: Path | None,
) -> WhisperModelLoadError:
    """Build a specific and friendly SSL certificate error message."""

    expected_path = model_path or model_cache_dir / f"{model_name}.pt"
    return WhisperModelLoadError(
        "DYANA could not download the Whisper model because HTTPS certificate validation failed.\n"
        f"Model: {model_name}\n"
        f"Expected local checkpoint path: {expected_path}\n"
        f"Whisper cache directory: {model_cache_dir}\n"
        f"Original SSL error: {error}\n"
        "This usually means your network uses a proxy or custom certificate authority.\n"
        "To keep DYANA usable, provide a local checkpoint with --asr-model-path or place "
        f"the model in {model_cache_dir}. You can also set WHISPER_MODEL_DIR or "
        "WHISPER_MODEL_PATH before running DYANA."
    )


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


def _resample_linear(
    waveform: np.ndarray,
    sample_rate_in: int,
    sample_rate_out: int,
) -> np.ndarray:
    """Resample a mono waveform with linear interpolation."""

    if sample_rate_in <= 0 or sample_rate_out <= 0:
        raise ValueError("sample rates must be positive.")
    if waveform.size == 0 or sample_rate_in == sample_rate_out:
        return np.asarray(waveform, dtype=np.float32)

    duration_seconds = waveform.shape[0] / float(sample_rate_in)
    target_num_samples = max(1, int(round(duration_seconds * sample_rate_out)))
    source_positions = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_num_samples, endpoint=False)
    resampled = np.interp(target_positions, source_positions, waveform)
    return np.asarray(resampled, dtype=np.float32)


def _convert_whisper_segments(
    whisper_segments: list[dict[str, Any]],
    *,
    chunk_start_time: float,
    chunk_end_time: float,
) -> list[TranscriptSegment]:
    """Convert Whisper segment payloads into DYANA transcript dataclasses."""

    transcript_segments: list[TranscriptSegment] = []
    for whisper_segment in whisper_segments:
        words: list[WordTimestamp] = []
        for word in whisper_segment.get("words", []):
            word_text = str(word.get("word", "")).strip()
            if not word_text:
                continue

            word_start_time = max(chunk_start_time, chunk_start_time + float(word["start"]))
            word_end_time = min(chunk_end_time, chunk_start_time + float(word["end"]))
            if word_end_time - word_start_time < MIN_INTERVAL_DURATION_SECONDS:
                continue

            words.append(
                WordTimestamp(
                    word=word_text,
                    start_time=word_start_time,
                    end_time=word_end_time,
                    confidence=(
                        None if word.get("probability") is None else float(word["probability"])
                    ),
                )
            )

        segment_start_time = chunk_start_time + float(whisper_segment.get("start", 0.0))
        segment_end_time = chunk_start_time + float(
            whisper_segment.get("end", whisper_segment.get("start", 0.0))
        )
        clipped_segment_start = max(chunk_start_time, segment_start_time)
        clipped_segment_end = min(chunk_end_time, segment_end_time)
        if clipped_segment_end - clipped_segment_start < MIN_INTERVAL_DURATION_SECONDS:
            continue

        transcript_segments.append(
            TranscriptSegment(
                start_time=clipped_segment_start,
                end_time=clipped_segment_end,
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
