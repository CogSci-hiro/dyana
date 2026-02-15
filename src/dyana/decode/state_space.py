"""Decoder state-space definitions and helpers."""

from __future__ import annotations

from typing import Dict, List

# ##########  Canonical state set  ##########

STATE_NAMES: List[str] = ["SIL", "A", "B", "OVL", "LEAK"]
STATE_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(STATE_NAMES)}


def num_states() -> int:
    """Return the number of base states."""

    return len(STATE_NAMES)


def state_index(name: str) -> int:
    """Return index for a state name."""

    if name not in STATE_TO_INDEX:
        raise ValueError(f"Unknown state '{name}'. Expected one of {STATE_NAMES}.")
    return STATE_TO_INDEX[name]


def state_name(index: int) -> str:
    """Return state name for an index."""

    if index < 0 or index >= len(STATE_NAMES):
        raise ValueError(f"State index {index} out of range 0..{len(STATE_NAMES)-1}.")
    return STATE_NAMES[index]


def all_state_indices() -> range:
    """Convenience range over state indices."""

    return range(len(STATE_NAMES))
