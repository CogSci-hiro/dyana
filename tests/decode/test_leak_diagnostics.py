from dyana.decode.ipu import count_ipu_starts_after_leak


def test_count_ipu_starts_after_leak() -> None:
    states = ["SIL", "LEAK", "LEAK", "A", "A", "SIL", "LEAK", "B", "B"]
    assert count_ipu_starts_after_leak(states) == 2
