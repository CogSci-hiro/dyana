import numpy as np

from dyana.eval.metrics import boundary_f1, framewise_iou, micro_ipus_per_min, speaker_switches_per_min, rapid_alternations
from dyana.decode.ipu import Segment


def test_boundary_f1_simple_match() -> None:
    ref = np.array([0.1, 0.5])
    hyp = np.array([0.11, 0.7])
    res = boundary_f1(ref, hyp, tol_s=0.05)
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 1


def test_framewise_iou_empty_union() -> None:
    assert framewise_iou(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool)) == 1.0
    assert framewise_iou(np.zeros(3, dtype=bool), np.array([1, 0, 0], dtype=bool)) == 0.0


def test_structural_metrics() -> None:
    ipus = [Segment(0.0, 0.1, "A"), Segment(0.5, 0.9, "A")]
    micro = micro_ipus_per_min(ipus, total_duration_s=60.0, max_duration_s=0.2)
    assert 0.9 < micro < 1.1  # about 1 per min

    states = ["SIL", "A", "A", "B", "B", "A", "A"]
    switches = speaker_switches_per_min(states, 0.01)
    assert switches > 0

    rapid = rapid_alternations(["A", "B", "A", "SIL"], 0.01)
    assert rapid == 1
