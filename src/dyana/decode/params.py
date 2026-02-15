"""Decoder tuning parameters for constrained transition scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeTuningParams:
    """Small bundle of decode tuning knobs."""

    speaker_switch_penalty: float = -6.0
    leak_entry_bias: float = -2.0
    ovl_transition_cost: float = -3.0
