from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from dyana.cli.commands import tune
from dyana.errors import PipelineError


def _write_manifest(path: Path) -> None:
    path.write_text(json.dumps([{"id": "easy_1", "tier": "easy", "audio_path": "x.wav", "ref_path": "x.npy"}]))


def _write_baseline(path: Path) -> None:
    payload = {
        "results": [
            {
                "id": "easy_1",
                "tier": "easy",
                "boundary_f1_20ms": 0.90,
                "boundary_f1_50ms": 0.95,
                "micro_ipus_per_min": 1.0,
                "switches_per_min": 2.0,
            }
        ],
        "metadata": {
            "params": {
                "speaker_switch_penalty": -5.0,
                "leak_entry_bias": -2.0,
                "ovl_transition_cost": -3.0,
            }
        },
        "summary": {},
    }
    path.write_text(json.dumps(payload))


def test_tune_writes_delta_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    baseline_path = tmp_path / "baseline.json"
    _write_manifest(manifest_path)
    _write_baseline(baseline_path)

    monkeypatch.setattr(
        tune,
        "evaluate_manifest",
        lambda *args, **kwargs: [
            {
                "id": "easy_1",
                "tier": "easy",
                "boundary_f1_20ms": 0.92,
                "boundary_f1_50ms": 0.96,
                "micro_ipus_per_min": 0.9,
                "switches_per_min": 1.8,
            }
        ],
    )

    args = argparse.Namespace(
        command="tune",
        manifest=str(manifest_path),
        baseline=str(baseline_path),
        out_dir=str(tmp_path / "out"),
        run_name="current",
        cache_dir=None,
        speaker_switch_penalty=None,
        leak_entry_bias=None,
        ovl_transition_cost=None,
        grid=False,
    )
    tune.run(args)

    assert (tmp_path / "out" / "current" / "delta.json").exists()
    assert (tmp_path / "out" / "current" / "delta.csv").exists()


def test_tune_guardrail_failure_raises_pipeline_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    baseline_path = tmp_path / "baseline.json"
    _write_manifest(manifest_path)
    _write_baseline(baseline_path)

    monkeypatch.setattr(
        tune,
        "evaluate_manifest",
        lambda *args, **kwargs: [
            {
                "id": "easy_1",
                "tier": "easy",
                "boundary_f1_20ms": 0.70,
                "boundary_f1_50ms": 0.80,
                "micro_ipus_per_min": 2.0,
                "switches_per_min": 4.0,
            }
        ],
    )

    args = argparse.Namespace(
        command="tune",
        manifest=str(manifest_path),
        baseline=str(baseline_path),
        out_dir=str(tmp_path / "out"),
        run_name="regressed",
        cache_dir=None,
        speaker_switch_penalty=None,
        leak_entry_bias=None,
        ovl_transition_cost=None,
        grid=False,
    )

    with pytest.raises(PipelineError):
        tune.run(args)

    delta_path = tmp_path / "out" / "regressed" / "delta.json"
    assert delta_path.exists()
    payload = json.loads(delta_path.read_text())
    assert payload["failed"] is True
