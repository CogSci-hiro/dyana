import pytest
from dyana.core.timebase import TimeBase

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
