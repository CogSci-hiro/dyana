import numpy as np
from dyana.decode import decoder, constraints, state_space


def test_viterbi_determinism() -> None:
    scores = decoder.random_scores(10, seed=123)
    path1 = decoder.decode_with_constraints(scores)
    path2 = decoder.decode_with_constraints(scores)
    assert path1 == path2


def test_scripted_blocks_decode_as_expected() -> None:
    blocks = [("A", 4), ("SIL", 3), ("B", 4)]
    scores = decoder.scripted_block_scores(blocks, margin=6.0)
    path = decoder.decode_with_constraints(scores)

    # Collapse contiguous segments to base for comparison
    segments = []
    for p in path:
        if not segments or segments[-1] != p:
            segments.append(p)
    assert segments[0] == "A"
    assert segments[1] == "SIL"
    assert segments[2] == "B"


def test_no_ping_pong_under_random_scores() -> None:
    min_durs = constraints.default_min_durations()
    min_durs["A"] = 2
    min_durs["B"] = 2
    scores = decoder.random_scores(40, seed=1)
    path = decoder.decode_with_constraints(scores, min_durations=min_durs)

    # Ensure A/B segments last at least min duration (no 1-frame ping-pong)
    current = path[0]
    run_len = 1
    for p in path[1:]:
        if p == current:
            run_len += 1
        else:
            if current in ("A", "B"):
                assert run_len >= min_durs[current]
            current = p
            run_len = 1


def test_leak_cannot_start_after_silence() -> None:
    T = 5
    S = state_space.num_states()
    scores = np.zeros((T, S))
    # frame 0 neutral
    # frame 1 strongly favors LEAK
    scores[1, state_space.state_index("LEAK")] = 5.0
    scores[1, state_space.state_index("A")] = 1.0
    scores[1, state_space.state_index("B")] = 1.0
    # rest neutral

    init = np.zeros(S)
    init[state_space.state_index("SIL")] = 0.0
    init[state_space.state_index("LEAK")] = -5.0  # encourage SIL start

    path = decoder.decode_with_constraints(scores, initial=init)
    assert "LEAK" not in path[:2]  # not allowed immediately after start/silence
