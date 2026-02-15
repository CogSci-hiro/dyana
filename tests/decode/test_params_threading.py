import numpy as np

from dyana.decode import constraints, decoder, state_space
from dyana.decode.params import DecodeTuningParams


def test_default_params_preserve_transition_matrix() -> None:
    base_default = constraints.base_transition_matrix()
    with_params = constraints.base_transition_matrix(tuning_params=DecodeTuningParams())
    assert np.allclose(base_default, with_params, equal_nan=True)


def test_higher_speaker_switch_penalty_reduces_switches() -> None:
    T = 120
    rng = np.random.default_rng(7)
    scores = rng.standard_normal((T, state_space.num_states()))
    path_lo = decoder.decode_with_constraints(
        scores,
        tuning_params=DecodeTuningParams(speaker_switch_penalty=-4.0, leak_entry_bias=-2.0, ovl_transition_cost=-3.0),
    )
    path_hi = decoder.decode_with_constraints(
        scores,
        tuning_params=DecodeTuningParams(speaker_switch_penalty=-10.0, leak_entry_bias=-2.0, ovl_transition_cost=-3.0),
    )

    def switches(path: list[str]) -> int:
        count = 0
        last = None
        for label in path:
            if label not in ("A", "B"):
                continue
            if last is not None and label != last:
                count += 1
            last = label
        return count

    assert switches(path_hi) <= switches(path_lo)
