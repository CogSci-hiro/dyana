"""Transition and duration constraints for the Axis 2 decoder."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Sequence, Tuple

from dyana.decode import state_space


# ---------- Default penalties (log-domain additive) ----------

STAY_REWARD: float = 0.0
GENERIC_SWITCH_PENALTY: float = -3.0
SPEAKER_SWITCH_PENALTY: float = -6.0  # A <-> B
SIL_EXIT_PENALTY: float = -1.0
SIL_ENTER_PENALTY: float = -0.5
LEAK_FORBID: float = -np.inf  # forbid SIL -> LEAK

# ---------- Default minimum durations (frames) ----------

MIN_IPU_FRAMES: int = 3  # A/B/OVL/LEAK
MIN_SIL_FRAMES: int = 2


def base_transition_matrix(states: Sequence[str] | None = None) -> np.ndarray:
    """Return the base (unexpanded) transition log-penalty matrix."""

    names = list(states) if states is not None else state_space.STATE_NAMES
    S = len(names)
    mat = np.full((S, S), GENERIC_SWITCH_PENALTY, dtype=float)

    for i in range(S):
        mat[i, i] = STAY_REWARD

    name_to_idx = {n: i for i, n in enumerate(names)}

    a = name_to_idx["A"]
    b = name_to_idx["B"]
    mat[a, b] = SPEAKER_SWITCH_PENALTY
    mat[b, a] = SPEAKER_SWITCH_PENALTY

    sil = name_to_idx["SIL"]
    leak = name_to_idx["LEAK"]
    mat[sil, leak] = LEAK_FORBID
    mat[sil, :] += SIL_EXIT_PENALTY
    mat[:, sil] += SIL_ENTER_PENALTY
    mat[sil, sil] = STAY_REWARD  # keep SIL self loop clean
    return mat


# ---------- Duration expansion ----------

ExpandedState = Tuple[str, int]  # (base_name, sub_idx)


def expand_state_space(
    min_durations: Dict[str, int],
    base_states: Sequence[str] | None = None,
    base_transition: np.ndarray | None = None,
) -> Tuple[List[ExpandedState], np.ndarray, List[str]]:
    """
    Expand base states into duration-enforcing substates.

    Returns
    -------
    expanded_states : list of (base, sub_idx)
    expanded_transition : ndarray (Sexp, Sexp)
    collapse_map : list mapping expanded index -> base name
    """

    base_names = list(base_states) if base_states is not None else state_space.STATE_NAMES
    base_trans = base_transition if base_transition is not None else base_transition_matrix(base_names)
    base_index: Dict[str, int] = {name: idx for idx, name in enumerate(base_names)}

    expanded_states: List[ExpandedState] = []
    for name in base_names:
        d = max(1, int(min_durations.get(name, 1)))
        for k in range(d):
            expanded_states.append((name, k))

    S_exp = len(expanded_states)
    trans = np.full((S_exp, S_exp), -np.inf, dtype=float)

    # Precompute first substate index for each base
    first_index: Dict[str, int] = {}
    for idx, (name, sub) in enumerate(expanded_states):
        if sub == 0:
            first_index[name] = idx

    # Fill transitions
    for i, (src_base, src_sub) in enumerate(expanded_states):
        d_src = max(1, int(min_durations.get(src_base, 1)))
        if src_sub < d_src - 1:
            # must stay within duration chain
            trans[i, i + 1] = STAY_REWARD
            continue

        # At final substate: allow transitions based on base matrix
        for j, (dst_base, dst_sub) in enumerate(expanded_states):
            if dst_sub != 0:
                continue  # only enter first substate of destination
            base_pen = base_trans[base_index[src_base], base_index[dst_base]]
            trans[i, j] = base_pen

        # staying in same base after duration satisfied
        trans[i, i] = STAY_REWARD

    collapse_map = [b for (b, _) in expanded_states]
    return expanded_states, trans, collapse_map


def default_min_durations() -> Dict[str, int]:
    """Default minimum durations per base state."""

    return {
        "SIL": MIN_SIL_FRAMES,
        "A": MIN_IPU_FRAMES,
        "B": MIN_IPU_FRAMES,
        "OVL": MIN_IPU_FRAMES,
        "LEAK": MIN_IPU_FRAMES,
    }
