import pytest
from dyana.core.timebase import CANONICAL_HOP_SECONDS, TimeBase


def test_canonical_timebase_has_expected_hop() -> None:
    tb = TimeBase.canonical()
    assert tb.hop_s == CANONICAL_HOP_SECONDS
    assert tb.hop_ms == 10.0


def test_canonical_frame_mapping_exact() -> None:
    tb = TimeBase.canonical()
    assert tb.frame_to_time(0) == 0.0
    assert tb.frame_to_time(1) == CANONICAL_HOP_SECONDS
    assert tb.frame_to_time(42) == 0.42
    assert tb.frame_to_time(10) == 0.10


def test_timebase_roundtrip_floor_semantics() -> None:
    tb = TimeBase(hop_s=0.01)

    t = 0.421
    i = tb.time_to_frame(t)
    assert i == 42
    assert tb.frame_to_time(i) <= t
    assert t < tb.frame_to_time(i + 1)

def test_num_frames_covers_duration() -> None:
    tb = TimeBase(hop_s=0.01)

    d = 3.7
    n = tb.num_frames(d)

    assert n == 370
    assert tb.frame_to_time(n - 1) < d
    assert d <= tb.frame_to_time(n)

def test_negative_inputs_raise() -> None:
    tb = TimeBase(hop_s=0.01)

    with pytest.raises(ValueError):
        _ = tb.frame_to_time(-1)

    with pytest.raises(ValueError):
        _ = tb.time_to_frame(-0.1)

    with pytest.raises(ValueError):
        _ = tb.num_frames(-1.0)


def test_frame_times_shape_and_values() -> None:
    tb = TimeBase(hop_s=0.02)
    out = tb.frame_times(4)
    assert out.tolist() == [0.0, 0.02, 0.04, 0.06]


def test_invalid_hop_and_frame_count_raise() -> None:
    with pytest.raises(ValueError):
        _ = TimeBase(hop_s=0.0)

    tb = TimeBase(hop_s=0.01)
    with pytest.raises(ValueError):
        _ = tb.frame_times(-1)


def test_require_canonical_raises_for_non_canonical() -> None:
    tb = TimeBase.from_hop_seconds(0.02)
    with pytest.raises(ValueError):
        tb.require_canonical()
