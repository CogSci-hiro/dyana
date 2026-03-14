"""Stereo-derived evidence tracks for channel bias and overlap heuristics."""

from __future__ import annotations

import numpy as np

from dyana.core.timebase import TimeBase
from dyana.evidence.base import EvidenceTrack
from dyana.evidence.bundle import EvidenceBundle


ENERGY_EPSILON: float = 1e-8
CORRELATION_STD_EPSILON: float = 1e-8
EXPECTED_STEREO_CHANNELS: int = 2


def _validate_stereo_audio(audio: np.ndarray) -> np.ndarray:
    """Validate and normalize stereo audio input to float32."""

    stereo = np.asarray(audio, dtype=np.float32)
    if stereo.ndim != 2:
        raise ValueError(f"Expected audio with shape (samples, channels), got ndim={stereo.ndim}.")
    if stereo.shape[1] != EXPECTED_STEREO_CHANNELS:
        raise ValueError(
            f"Expected stereo audio with {EXPECTED_STEREO_CHANNELS} channels, got shape {stereo.shape}."
        )
    return stereo


def _frame_count(n_samples: int, frame_hop: int) -> int:
    """Return the number of complete frames available at the requested hop."""

    if frame_hop <= 0:
        raise ValueError("frame_hop must be positive.")
    n_frames = int(n_samples // frame_hop)
    if n_frames <= 0:
        raise ValueError("Audio is too short for one complete frame.")
    return n_frames


def _resolve_audio_and_frame_hop(
    audio: np.ndarray | tuple[np.ndarray, int],
    timebase: TimeBase,
) -> tuple[np.ndarray, int]:
    """Resolve stereo audio and frame hop in samples for bundle construction."""

    if isinstance(audio, tuple):
        stereo, sample_rate = audio
        frame_hop = int(round(sample_rate * timebase.hop_s))
        if frame_hop <= 0:
            raise ValueError("timebase hop is too small for the provided sample rate.")
        return _validate_stereo_audio(stereo), frame_hop

    stereo = _validate_stereo_audio(audio)
    if timebase.n_frames is None:
        raise ValueError("timebase.n_frames is required when sample_rate is not provided.")
    if stereo.shape[0] % timebase.n_frames != 0:
        raise ValueError(
            "Cannot infer frame_hop from audio shape and timebase.n_frames; provide (audio, sample_rate) instead."
        )
    frame_hop = stereo.shape[0] // timebase.n_frames
    if frame_hop <= 0:
        raise ValueError("Inferred frame_hop must be positive.")
    return stereo, frame_hop


def compute_channel_energy(audio: np.ndarray, frame_hop: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame RMS energy for stereo channels.

    Parameters
    ----------
    audio
        Stereo waveform with shape ``(samples, channels)`` and exactly two
        channels ordered as left then right.
    frame_hop
        Frame size in samples. DYANA uses the canonical 10 ms hop, so callers
        should derive this from the active :class:`~dyana.core.timebase.TimeBase`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two float32 arrays ``(energy_left, energy_right)`` with shape ``(T,)``,
        where each value is

        ``sqrt(mean(frame**2))``.

    Notes
    -----
    In the decoder fusion stage these energies are intended as raw score-like
    tracks rather than calibrated probabilities.
    """

    stereo = _validate_stereo_audio(audio)
    n_frames = _frame_count(stereo.shape[0], frame_hop)
    trimmed = stereo[: n_frames * frame_hop]
    framed = trimmed.reshape(n_frames, frame_hop, EXPECTED_STEREO_CHANNELS)
    rms = np.sqrt(np.mean(framed * framed, axis=1, dtype=np.float32)).astype(np.float32)
    return rms[:, 0].astype(np.float32), rms[:, 1].astype(np.float32)


def compute_energy_ratio(left_energy: np.ndarray, right_energy: np.ndarray) -> np.ndarray:
    """
    Compute a log energy ratio between stereo channels.

    Parameters
    ----------
    left_energy
        Left-channel RMS energy per frame.
    right_energy
        Right-channel RMS energy per frame.

    Returns
    -------
    np.ndarray
        Float32 array ``ratio`` with shape ``(T,)`` defined as

        ``ratio[t] = log((left_energy[t] + eps) / (right_energy[t] + eps))``

        where ``eps`` is :data:`ENERGY_EPSILON`.

    Notes
    -----
    Positive values indicate left-channel dominance and therefore softly favor
    speaker A. Negative values indicate right-channel dominance and softly favor
    speaker B. These values are designed to be consumed later as an uncalibrated
    decoder-fusion feature.
    """

    left = np.asarray(left_energy, dtype=np.float32)
    right = np.asarray(right_energy, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"Energy arrays must have matching shape, got {left.shape} and {right.shape}.")
    ratio = np.log((left + ENERGY_EPSILON) / (right + ENERGY_EPSILON))
    return ratio.astype(np.float32)


def compute_cross_channel_correlation(audio: np.ndarray, frame_hop: int) -> np.ndarray:
    """
    Compute per-frame Pearson correlation between left and right channels.

    Parameters
    ----------
    audio
        Stereo waveform with shape ``(samples, 2)``.
    frame_hop
        Frame size in samples, typically corresponding to the canonical 10 ms
        timebase hop.

    Returns
    -------
    np.ndarray
        Float32 array ``corr`` with shape ``(T,)`` and values clipped to
        ``[-1, 1]``.

    Notes
    -----
    Each frame is centered independently per channel, scaled by its standard
    deviation, and then scored with the Pearson correlation coefficient. High
    correlation paired with high energy suggests overlap or channel leakage,
    while low correlation paired with high energy is more consistent with a
    single active speaker localized to one side.
    """

    stereo = _validate_stereo_audio(audio)
    n_frames = _frame_count(stereo.shape[0], frame_hop)
    trimmed = stereo[: n_frames * frame_hop]
    framed = trimmed.reshape(n_frames, frame_hop, EXPECTED_STEREO_CHANNELS).astype(np.float32, copy=False)

    left = framed[:, :, 0]
    right = framed[:, :, 1]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)

    left_std = left_centered.std(axis=1)
    right_std = right_centered.std(axis=1)
    denom = left_std * right_std

    corr = np.zeros(n_frames, dtype=np.float32)
    valid = denom > CORRELATION_STD_EPSILON
    if np.any(valid):
        left_norm = left_centered[valid] / left_std[valid, None]
        right_norm = right_centered[valid] / right_std[valid, None]
        corr[valid] = np.mean(left_norm * right_norm, axis=1, dtype=np.float32)

    degenerate = np.flatnonzero(~valid)
    for index in degenerate:
        if np.allclose(left[index], right[index], atol=1e-7, rtol=1e-5, equal_nan=False):
            corr[index] = 1.0

    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def compute_stereo_evidence(audio: np.ndarray | tuple[np.ndarray, int], timebase: TimeBase) -> EvidenceBundle:
    """
    Compute stereo-derived evidence tracks on the provided timebase.

    Parameters
    ----------
    audio
        Stereo waveform with shape ``(samples, 2)`` or a tuple
        ``(waveform, sample_rate)``. Passing the sample rate is the preferred
        form because it lets the function derive the 10 ms frame hop exactly.
    timebase
        Canonical DYANA timebase describing the desired frame hop and frame
        count.

    Returns
    -------
    EvidenceBundle
        Bundle containing four score-valued tracks:

        ``stereo_energy_left``
            Left-channel RMS energy.
        ``stereo_energy_right``
            Right-channel RMS energy.
        ``stereo_ratio``
            Log energy ratio favoring speaker A when positive and speaker B when
            negative.
        ``stereo_corr``
            Cross-channel similarity score derived from per-frame Pearson
            correlation.

    Notes
    -----
    The energy ratio is defined as

    ``log((E_left + eps) / (E_right + eps))``.

    The correlation track is the per-frame Pearson coefficient between the two
    channels after centering and scaling. In later decoder fusion this bundle is
    intended to provide deterministic soft cues for channel bias and
    overlap-versus-leakage interpretation without changing the decoder itself.

    Usage example
    -------------
    >>> tb = TimeBase.canonical()
    >>> bundle = compute_stereo_evidence((stereo_audio, 16000), tb)
    >>> ratio_track = bundle.get("stereo_ratio")
    >>> corr_track = bundle.get("stereo_corr")
    """

    timebase.require_canonical()
    stereo, frame_hop = _resolve_audio_and_frame_hop(audio, timebase)

    left_energy, right_energy = compute_channel_energy(stereo, frame_hop=frame_hop)
    ratio = compute_energy_ratio(left_energy, right_energy)
    corr = compute_cross_channel_correlation(stereo, frame_hop=frame_hop)

    n_frames = left_energy.shape[0]
    if timebase.n_frames is not None and timebase.n_frames != n_frames:
        raise ValueError(f"Stereo evidence length {n_frames} does not match timebase.n_frames={timebase.n_frames}.")

    track_timebase = TimeBase.canonical(n_frames=n_frames)
    bundle = EvidenceBundle(timebase=track_timebase)
    bundle.add_track(
        "stereo_energy_left",
        EvidenceTrack(
            name="stereo_energy_left",
            timebase=track_timebase,
            values=left_energy,
            semantics="score",
            metadata={"description": "Left-channel RMS energy per frame.", "channel": "left"},
        ),
    )
    bundle.add_track(
        "stereo_energy_right",
        EvidenceTrack(
            name="stereo_energy_right",
            timebase=track_timebase,
            values=right_energy,
            semantics="score",
            metadata={"description": "Right-channel RMS energy per frame.", "channel": "right"},
        ),
    )
    bundle.add_track(
        "stereo_ratio",
        EvidenceTrack(
            name="stereo_ratio",
            timebase=track_timebase,
            values=ratio,
            semantics="score",
            metadata={
                "description": "Log left/right energy ratio; positive favors speaker A, negative favors speaker B.",
                "formula": "log((left_energy + eps) / (right_energy + eps))",
            },
        ),
    )
    bundle.add_track(
        "stereo_corr",
        EvidenceTrack(
            name="stereo_corr",
            timebase=track_timebase,
            values=corr,
            semantics="score",
            metadata={
                "description": "Per-frame Pearson correlation between stereo channels.",
                "interpretation": "Higher values indicate stronger cross-channel similarity and possible overlap or leakage.",
            },
        ),
    )
    return bundle
