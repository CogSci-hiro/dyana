import json
from pathlib import Path

import pytest

from dyana.eval.tuning import compute_delta_report, write_delta_report


def test_delta_report_identical_inputs_are_zero(tmp_path: Path) -> None:
    payload = {
        "metadata": {"params": {"speaker_switch_penalty": -6.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0}},
        "results": [
            {
                "id": "x1",
                "tier": "easy",
                "boundary_f1_20ms": 0.8,
                "boundary_f1_50ms": 0.9,
                "micro_ipus_per_min": 2.0,
                "switches_per_min": 5.0,
                "status": "ok",
            }
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(payload))
    report = compute_delta_report(
        payload,
        payload,
        params={"speaker_switch_penalty": -6.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0},
        baseline_path=baseline_path,
    )
    row = report["rows"][0]
    assert row["boundary_f1_20ms_delta"] == pytest.approx(0.0)
    assert row["micro_ipus_per_min_delta"] == pytest.approx(0.0)
    assert report["failed"] is False


def test_delta_report_computation_and_writes(tmp_path: Path) -> None:
    baseline_payload = {
        "results": [
            {
                "id": "x1",
                "tier": "easy",
                "boundary_f1_20ms": 0.8,
                "boundary_f1_50ms": 0.9,
                "micro_ipus_per_min": 2.0,
                "switches_per_min": 5.0,
            }
        ]
    }
    current_payload = {
        "results": [
            {
                "id": "x1",
                "tier": "easy",
                "boundary_f1_20ms": 0.85,
                "boundary_f1_50ms": 0.92,
                "micro_ipus_per_min": 1.8,
                "switches_per_min": 4.0,
            }
        ]
    }

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload))
    report = compute_delta_report(
        baseline_payload,
        current_payload,
        params={"speaker_switch_penalty": -6.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0},
        baseline_path=baseline_path,
    )

    assert report["rows"][0]["boundary_f1_20ms_delta"] == pytest.approx(0.05)
    assert report["rows"][0]["micro_ipus_per_min_delta"] == pytest.approx(-0.2)
    write_delta_report(report, tmp_path)
    assert (tmp_path / "delta.json").exists()
    assert (tmp_path / "delta.csv").exists()


def test_delta_report_flags_unexplained_when_params_identical(tmp_path: Path) -> None:
    baseline_payload = {
        "metadata": {"params": {"speaker_switch_penalty": -6.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0}},
        "results": [
            {
                "id": "hard_1",
                "tier": "hard",
                "boundary_f1_20ms": 0.7,
                "boundary_f1_50ms": 0.8,
                "micro_ipus_per_min": 3.0,
                "switches_per_min": 2.0,
                "status": "ok",
            }
        ],
    }
    current_payload = {
        "metadata": {"params": {"speaker_switch_penalty": -6.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0}},
        "results": [
            {
                "id": "hard_1",
                "tier": "hard",
                "boundary_f1_20ms": 0.75,
                "boundary_f1_50ms": 0.82,
                "micro_ipus_per_min": 2.5,
                "switches_per_min": 2.1,
                "status": "ok",
            }
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload))
    report = compute_delta_report(
        baseline_payload,
        current_payload,
        params={"speaker_switch_penalty": -6.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0},
        baseline_path=baseline_path,
    )
    assert report["failed"] is True
    assert any("UNEXPLAINED" in text for text in report["failures"])


def test_delta_report_param_change_without_metric_change_is_explained(tmp_path: Path) -> None:
    baseline_payload = {
        "metadata": {"params": {"speaker_switch_penalty": -6.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0}},
        "results": [
            {
                "id": "easy_1",
                "tier": "easy",
                "boundary_f1_20ms": 0.9,
                "boundary_f1_50ms": 0.95,
                "micro_ipus_per_min": 1.0,
                "switches_per_min": 1.0,
                "status": "ok",
            }
        ],
    }
    current_payload = {
        "metadata": {"params": {"speaker_switch_penalty": -7.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0}},
        "results": [
            {
                "id": "easy_1",
                "tier": "easy",
                "boundary_f1_20ms": 0.9,
                "boundary_f1_50ms": 0.95,
                "micro_ipus_per_min": 1.0,
                "switches_per_min": 1.0,
                "status": "ok",
            }
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload))
    report = compute_delta_report(
        baseline_payload,
        current_payload,
        params={"speaker_switch_penalty": -7.0, "leak_entry_bias": -2.0, "ovl_transition_cost": -3.0},
        baseline_path=baseline_path,
    )
    assert report["failed"] is False
