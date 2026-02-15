import pytest

from dyana.decode import state_space


def test_state_order_and_indices() -> None:
    expected = ["SIL", "A", "B", "OVL", "LEAK"]
    assert state_space.STATE_NAMES == expected
    for i, name in enumerate(expected):
        assert state_space.state_index(name) == i
        assert state_space.state_name(i) == name


def test_state_index_bounds() -> None:
    with pytest.raises(ValueError):
        state_space.state_index("UNKNOWN")
    with pytest.raises(ValueError):
        state_space.state_name(len(state_space.STATE_NAMES))
