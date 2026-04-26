"""Evidence fusion from EvidenceBundle to base-state log scores."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dyana.evidence.bundle import EvidenceBundle
from dyana.evidence.base import EvidenceTrack
from dyana.decode import state_space
from dyana.decode.params import DecodeTuningParams

# ---------- Constants ----------

LOG_EPS: float = 1e-6
W_DIAR: float = 1.0
W_OVL: float = 1.5
W_LEAK: float = 1.0
W_LEAK_SIL_BIAS: float = 0.5
LEAK_BASELINE_PENALTY: float = -3.0
W_PRIOR: float = 0.4
OVL_BONUS: float = 0.4
W_OVL_PROXY: float = 0.5
HIGH_RECALL_SILENCE_SCORE_DELTA: float = -1.0
LEAKAGE_TRACK_NAME: str = "leakage_likelihood"
OVERLAP_PROXY_TRACK_NAME: str = "overlap_proxy"
PYANNOTE_SPEECH_TRACK_NAME: str = "pyannote_speech"


def _require_prob_or_logit(track: EvidenceTrack, name: str) -> NDArray[np.floating]:
    if track.semantics == "probability":
        return np.asarray(track.values, dtype=float)
    if track.semantics == "logit":
        return 1.0 / (1.0 + np.exp(-np.asarray(track.values, dtype=float)))
    raise ValueError(f"Track '{name}' must have semantics probability or logit, got {track.semantics}.")


def _log_prob(p: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.log(np.clip(p, LOG_EPS, 1.0 - LOG_EPS))


def _log_not_prob(p: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.log(np.clip(1.0 - p, LOG_EPS, 1.0 - LOG_EPS))


def _score_track_to_probability(track: EvidenceTrack | None, *, default: float, name: str) -> NDArray[np.floating] | None:
    if track is None:
        return None
    values = np.asarray(track.values, dtype=float)
    if track.semantics in ("probability", "logit"):
        return _require_prob_or_logit(track, name)
    if track.semantics != "score":
        raise ValueError(f"Track '{name}' has unsupported semantics {track.semantics}.")
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    if high - low < 1e-6:
        return np.full(values.shape[0], default, dtype=float)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _blend_speech_evidence(bundle: EvidenceBundle, params: DecodeTuningParams, T: int) -> NDArray[np.floating]:
    weighted_sum = np.zeros(T, dtype=float)
    total_weight = 0.0

    sources = (
        ("vad", bundle.get("vad"), params.speech_weight_vad, 0.5),
        (PYANNOTE_SPEECH_TRACK_NAME, bundle.get(PYANNOTE_SPEECH_TRACK_NAME), params.speech_weight_pyannote, 0.0),
        ("energy_smooth", bundle.get("energy_smooth"), params.speech_weight_energy, 0.0),
        ("voiced_soft", bundle.get("voiced_soft"), params.speech_weight_voiced, 0.0),
    )

    for name, track, weight, default in sources:
        if track is None or weight <= 0.0:
            continue
        probs = _score_track_to_probability(track, default=default, name=name)
        if probs is None:
            continue
        weighted_sum += weight * probs
        total_weight += weight

    if total_weight <= 0.0:
        return np.full(T, 0.5, dtype=float)
    return np.clip(weighted_sum / total_weight, 0.0, 1.0)


def _check_timebases(bundle: EvidenceBundle) -> int:
    tb = bundle.timebase
    tb.require_canonical()
    n_frames: int | None = tb.n_frames
    for name, track in bundle.items():
        if abs(track.timebase.hop_s - tb.hop_s) > 1e-12:
            raise ValueError(
                f"Track '{name}' hop {track.timebase.hop_s} does not match bundle hop {tb.hop_s}."
            )
        if n_frames is None:
            n_frames = track.T
        if track.T != n_frames:
            raise ValueError(
                f"Track '{name}' length {track.T} mismatches bundle length {n_frames}."
            )
    if n_frames is None:
        raise ValueError("EvidenceBundle is empty; cannot fuse without tracks.")
    return n_frames


def fuse_bundle_to_scores(
    bundle: EvidenceBundle,
    *,
    tuning_params: DecodeTuningParams | None = None,
) -> NDArray[np.floating]:
    """
    Convert an EvidenceBundle into log-scores over base states.

    Notes
    -----
    Balanced mode preserves the default silence handling.
    High-recall mode is intended for pre-transcription segmentation and reduces
    silence preference so short energy dips are less likely to split speech.

    Returns
    -------
    scores : ndarray
        Shape (T, num_states) in log space.
    """

    params = tuning_params or DecodeTuningParams()
    T = _check_timebases(bundle)
    S = state_space.num_states()
    scores = np.zeros((T, S), dtype=float)

    # Optional tracks
    diar_a = bundle.get("diar_a")
    diar_b = bundle.get("diar_b")
    leak = bundle.get(LEAKAGE_TRACK_NAME)
    if leak is None:
        leak = bundle.get("leak")
    prior = bundle.get("prior_ab")
    overlap_proxy = bundle.get(OVERLAP_PROXY_TRACK_NAME)

    # Speech / silence
    p_speech = _blend_speech_evidence(bundle, params, T)
    log_speech = _log_prob(p_speech)
    log_nonspeech = _log_not_prob(p_speech)

    # Diarization
    if diar_a is not None:
        p_a = _require_prob_or_logit(diar_a, "diar_a")
    else:
        p_a = np.full(T, 0.5, dtype=float)
    if diar_b is not None:
        p_b = _require_prob_or_logit(diar_b, "diar_b")
    else:
        p_b = np.full(T, 0.5, dtype=float)
    log_pa = _log_prob(p_a)
    log_pb = _log_prob(p_b)

    # Prior bias
    prior_offset_a = prior_offset_b = 0.0
    if prior is not None:
        if prior.semantics != "score":
            raise ValueError("prior_ab track must use semantics='score' (additive log offset).")
        vals = np.asarray(prior.values, dtype=float)
        if vals.ndim == 1:
            if vals.shape[0] == 2:
                prior_offset_a = float(vals[0])
                prior_offset_b = float(vals[1])
            elif vals.shape[0] == 1:
                prior_offset_a = prior_offset_b = float(vals[0])
            else:
                raise ValueError("prior_ab 1-D values must be length 1 or 2.")
        elif vals.ndim == 2 and vals.shape[1] == 2:
            if vals.shape[0] != T:
                raise ValueError(f"prior_ab length {vals.shape[0]} mismatches bundle length {T}.")
            prior_offset_a = vals[:, 0]
            prior_offset_b = vals[:, 1]
        else:
            raise ValueError("prior_ab must have shape (T,2) or (2,) for A/B offsets.")

    # Leak
    if leak is not None:
        p_leak = _require_prob_or_logit(leak, LEAKAGE_TRACK_NAME if leak.name == LEAKAGE_TRACK_NAME else "leak")
        log_leak = _log_prob(p_leak)
    else:
        # Missing track should not crash decoding; keep conservative baseline.
        log_leak = np.zeros(T, dtype=float)

    if overlap_proxy is not None:
        p_ovl_proxy = _require_prob_or_logit(overlap_proxy, OVERLAP_PROXY_TRACK_NAME)
        log_ovl_proxy = _log_prob(p_ovl_proxy)
    else:
        log_ovl_proxy = np.zeros(T, dtype=float)

    idx_sil = state_space.state_index("SIL")
    idx_a = state_space.state_index("A")
    idx_b = state_space.state_index("B")
    idx_ovl = state_space.state_index("OVL")
    idx_leak = state_space.state_index("LEAK")

    silence_score = log_nonspeech + params.silence_bias - params.none_when_speech_penalty * p_speech
    if params.ipu_detection_mode == "high_recall":
        silence_score += HIGH_RECALL_SILENCE_SCORE_DELTA

    scores[:, idx_sil] += silence_score
    scores[:, idx_a] += (
        log_speech
        + params.speech_exists_to_single_speaker_bonus * p_speech
        + W_DIAR * log_pa
        + W_PRIOR * prior_offset_a
    )
    scores[:, idx_b] += (
        log_speech
        + params.speech_exists_to_single_speaker_bonus * p_speech
        + W_DIAR * log_pb
        + W_PRIOR * prior_offset_b
    )
    scores[:, idx_ovl] += (
        log_speech
        + W_OVL * (log_pa + log_pb)
        + W_OVL_PROXY * log_ovl_proxy
        + OVL_BONUS
    )
    # LEAK favors leakage evidence and silence-adjacent behavior.
    scores[:, idx_leak] += (
        W_LEAK * log_leak
        + W_LEAK_SIL_BIAS * log_nonspeech
        + LEAK_BASELINE_PENALTY
    )

    return scores
