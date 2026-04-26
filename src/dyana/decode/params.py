"""Decoder tuning parameters for constrained transition scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


ProfileName = Literal["default", "recall_first"]


@dataclass(frozen=True)
class DecodeTuningParams:
    """
    Small bundle of decode tuning knobs.

    Notes
    -----
    ``ovl_transition_cost`` is kept for backward compatibility and acts as a
    fallback when explicit OVL edge costs are not provided.

    ``ipu_detection_mode="balanced"`` preserves the current decoder behavior.
    ``ipu_detection_mode="high_recall"`` biases the system toward speech
    coverage for pre-transcription IPU segmentation, even if that slightly
    over-extends IPUs across silence-like dips.
    """

    speaker_switch_penalty: float = -6.0
    leak_entry_bias: float = -2.0
    ovl_transition_cost: float = -3.0
    a_to_ovl_cost: float | None = None
    b_to_ovl_cost: float | None = None
    ovl_to_a_cost: float | None = None
    ovl_to_b_cost: float | None = None
    ipu_detection_mode: Literal["balanced", "high_recall"] = "balanced"
    silence_bias: float = 0.0
    merge_silence_gap_ms: float = 400.0
    speech_weight_vad: float = 1.0
    speech_weight_pyannote: float = 0.0
    speech_weight_energy: float = 0.35
    speech_weight_voiced: float = 0.25
    none_when_speech_penalty: float = 1.2
    speech_exists_to_single_speaker_bonus: float = 0.15
    speech_evidence_threshold: float = 0.6
    profile: ProfileName = "default"

    @classmethod
    def for_profile(cls, profile: str) -> "DecodeTuningParams":
        """Construct a tuning bundle for a named profile.

        Usage example
        -------------
        >>> params = DecodeTuningParams.for_profile("recall-first")
        """

        normalized = profile.replace("-", "_").strip().lower()
        if normalized in ("default", "balanced"):
            return cls(profile="default")
        if normalized == "recall_first":
            return cls(
                ipu_detection_mode="high_recall",
                silence_bias=-0.35,
                merge_silence_gap_ms=500.0,
                speech_weight_vad=0.05,
                speech_weight_pyannote=1.35,
                speech_weight_energy=0.95,
                speech_weight_voiced=0.45,
                none_when_speech_penalty=2.4,
                speech_exists_to_single_speaker_bonus=0.25,
                speech_evidence_threshold=0.55,
                profile="recall_first",
            )
        raise ValueError(f"Unknown decode profile '{profile}'.")

    def resolved_ovl_costs(self) -> Tuple[float, float, float, float]:
        """Return explicit OVL transition costs in A->OVL, B->OVL, OVL->A, OVL->B order."""

        return (
            self.ovl_transition_cost if self.a_to_ovl_cost is None else self.a_to_ovl_cost,
            self.ovl_transition_cost if self.b_to_ovl_cost is None else self.b_to_ovl_cost,
            self.ovl_transition_cost if self.ovl_to_a_cost is None else self.ovl_to_a_cost,
            self.ovl_transition_cost if self.ovl_to_b_cost is None else self.ovl_to_b_cost,
        )
