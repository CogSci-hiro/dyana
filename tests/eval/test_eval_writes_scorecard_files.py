from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

pytest.importorskip("webrtcvad")

from dyana.cli.commands import eval as eval_cmd


def test_eval_command_writes_scorecard_files(tmp_path: Path) -> None:
    manifest = [
        {
            "id": "syn_leak_1",
            "tier": "synthetic",
            "scenario": "leakage_stress",
            "audio_path": None,
            "ref_path": None,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        command="eval",
        manifest=str(manifest_path),
        out_dir=str(out_dir),
        run_name="baseline",
        cache_dir=None,
    )
    eval_cmd.run(args)

    scorecard_json = out_dir / "baseline" / "scorecard.json"
    scorecard_csv = out_dir / "baseline" / "scorecard.csv"
    assert scorecard_json.exists()
    assert scorecard_csv.exists()
    payload = json.loads(scorecard_json.read_text())
    assert "results" in payload
    assert "summary" in payload
