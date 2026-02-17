"""Decoder tuning parameters for constrained transition scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DecodeTuningParams:
    """
    Small bundle of decode tuning knobs.

    Notes
    -----
    ``ovl_transition_cost`` is kept for backward compatibility and acts as a
    fallback when explicit OVL edge costs are not provided.
    """

    speaker_switch_penalty: float = -6.0
    leak_entry_bias: float = -2.0
    ovl_transition_cost: float = -3.0
    a_to_ovl_cost: float | None = None
    b_to_ovl_cost: float | None = None
    ovl_to_a_cost: float | None = None
    ovl_to_b_cost: float | None = None

    def resolved_ovl_costs(self) -> Tuple[float, float, float, float]:
        """Return explicit OVL transition costs in A->OVL, B->OVL, OVL->A, OVL->B order."""

        return (
            self.ovl_transition_cost if self.a_to_ovl_cost is None else self.a_to_ovl_cost,
            self.ovl_transition_cost if self.b_to_ovl_cost is None else self.b_to_ovl_cost,
            self.ovl_transition_cost if self.ovl_to_a_cost is None else self.ovl_to_a_cost,
            self.ovl_transition_cost if self.ovl_to_b_cost is None else self.ovl_to_b_cost,
        )
