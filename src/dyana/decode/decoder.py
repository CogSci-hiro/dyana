"""Structured decoder with constrained Viterbi over the Axis 2 state set."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dyana.decode import constraints, state_space


# ---------- Viterbi core ----------


def viterbi_decode(
    scores: np.ndarray,
    transition: np.ndarray,
    initial: Optional[np.ndarray] = None,
) -> Tuple[List[int], float]:
    """
    Run a standard Viterbi dynamic program in log space.

    Parameters
    ----------
    scores
        Emission log-scores, shape (T, S).
    transition
        Transition log-penalties, shape (S, S) with src on rows, dst on cols.
    initial
        Initial log-score per state, shape (S,). If None, zeros are used.

    Returns
    -------
    path
        Argmax state indices of length T.
    path_score
        Log-score of the best path.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape (T, S)")
    T, S = scores.shape
    if transition.shape != (S, S):
        raise ValueError(f"transition must have shape ({S}, {S})")

    init = np.zeros(S, dtype=float) if initial is None else np.asarray(initial, dtype=float)
    if init.shape != (S,):
        raise ValueError(f"initial must have shape ({S},)")

    dp = np.empty((T, S), dtype=float)
    bp = np.empty((T, S), dtype=int)

    dp[0] = init + scores[0]
    bp[0] = -1

    for t in range(1, T):
        prev = dp[t - 1][:, None] + transition  # shape (S, S)
        # deterministic tie-break: argmax picks lowest index first
        bp[t] = np.argmax(prev, axis=0)
        dp[t] = prev[bp[t], np.arange(S)] + scores[t]

    last_state = int(np.argmax(dp[-1]))
    path_score = float(dp[-1, last_state])

    path = [0] * T
    path[-1] = last_state
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(bp[t, path[t]])

    return path, path_score


# ---------- Constraint-aware decoding ----------


def expand_scores(base_scores: np.ndarray, expanded_states: Sequence[constraints.ExpandedState]) -> np.ndarray:
    """Copy base-state scores onto expanded substates."""

    T, S = base_scores.shape
    expanded = np.empty((T, len(expanded_states)), dtype=float)
    for j, (base, _) in enumerate(expanded_states):
        expanded[:, j] = base_scores[:, state_space.state_index(base)]
    return expanded


def decode_with_constraints(
    log_scores: np.ndarray,
    *,
    min_durations: Optional[Dict[str, int]] = None,
    transition: Optional[np.ndarray] = None,
    initial: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Decode a path respecting transition penalties and minimum durations.

    Parameters
    ----------
    log_scores
        Emission log-scores on base states, shape (T, S_base).
    min_durations
        Optional overrides for minimum durations per base state.
    transition
        Optional base transition matrix; if None the default is used.
    initial
        Optional initial log-scores.

    Returns
    -------
    path_names
        Decoded base-state names of length T.
    """

    base_names = state_space.STATE_NAMES
    if log_scores.shape[1] != len(base_names):
        raise ValueError(f"log_scores second dim must be {len(base_names)} (base states)")

    min_durs = min_durations or constraints.default_min_durations()
    base_trans = transition if transition is not None else constraints.base_transition_matrix(base_names)
    expanded_states, expanded_trans, collapse_map = constraints.expand_state_space(
        min_durs, base_states=base_names, base_transition=base_trans
    )
    expanded_scores = expand_scores(log_scores, expanded_states)

    # initial expanded: map base initial to first substate only
    if initial is None:
        init_base = np.zeros(len(base_names), dtype=float)
        # LEAK cannot initiate IPUs at sequence start.
        init_base[state_space.state_index("LEAK")] = -np.inf
    else:
        init_base = np.asarray(initial, dtype=float)
        if init_base.shape != (len(base_names),):
            raise ValueError(f"initial must have shape ({len(base_names)},)")
    init_expanded = np.full(len(expanded_states), -np.inf, dtype=float)
    for idx, (base, sub) in enumerate(expanded_states):
        if sub == 0:
            init_expanded[idx] = init_base[state_space.state_index(base)]

    path_idx, _ = viterbi_decode(expanded_scores, expanded_trans, initial=init_expanded)
    return [collapse_map[i] for i in path_idx]


def decode_diagnostics(states: Sequence[str]) -> Dict[str, int]:
    """
    Compute deterministic counters from decoded base-state sequence.

    Returns
    -------
    dict
        Contains ``ipu_start_after_leak_count``.
    """

    ipu_start_after_leak_count = 0
    if not states:
        return {"ipu_start_after_leak_count": 0}

    run_starts: List[int] = [0]
    for idx in range(1, len(states)):
        if states[idx] != states[idx - 1]:
            run_starts.append(idx)

    for run_start in run_starts:
        label = states[run_start]
        if label == "SIL":
            continue
        prev_idx = run_start - 1
        if prev_idx >= 0 and states[prev_idx] == "LEAK":
            ipu_start_after_leak_count += 1

    return {"ipu_start_after_leak_count": ipu_start_after_leak_count}


# ---------- Deterministic evidence helpers ----------


def scripted_block_scores(blocks: Sequence[Tuple[str, int]], margin: float = 5.0) -> np.ndarray:
    """
    Build deterministic scores where each block favors a given state.

    Parameters
    ----------
    blocks
        Sequence of (state_name, length) blocks.
    margin
        Positive margin added to the favored state.
    """

    total = sum(length for _, length in blocks)
    S = state_space.num_states()
    scores = np.zeros((total, S), dtype=float)
    t = 0
    for name, length in blocks:
        idx = state_space.state_index(name)
        for _ in range(length):
            scores[t, idx] += margin
            t += 1
    return scores


def random_scores(T: int, seed: int = 0) -> np.ndarray:
    """Deterministic random log-scores for testing."""

    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, state_space.num_states()))
