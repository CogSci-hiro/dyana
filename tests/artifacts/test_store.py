from __future__ import annotations

import json
from pathlib import Path

from dyana.artifacts.metadata import ArtifactMetadata
from dyana.artifacts.store import ArtifactStore


def test_store_save_load_roundtrip(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path / "artifacts")
    metadata = ArtifactMetadata(
        step="transcription",
        inputs={"audio": "abc123"},
        parameters={"model": "tiny"},
        timestamp=1.23,
        code_version="test-version",
    )

    artifact = store.save_artifact(
        name="transcription",
        obj={"transcript": "hello"},
        metadata=metadata,
    )

    assert store.artifact_exists(artifact.hash) is True
    assert store.load_artifact(artifact.hash) == {"transcript": "hello"}

    metadata_path = artifact.path.with_suffix(".json")
    saved_metadata = json.loads(metadata_path.read_text())
    assert saved_metadata["step"] == "transcription"
    assert saved_metadata["inputs"] == {"audio": "abc123"}
    assert saved_metadata["parameters"] == {"model": "tiny"}


def test_store_reuses_same_step_hash_for_same_inputs(tmp_path: Path) -> None:
    store = ArtifactStore(root=tmp_path / "artifacts")
    metadata = ArtifactMetadata(
        step="alignment",
        inputs={"audio": "input-hash", "transcript": "tx-hash"},
        parameters={"beam": 4},
        timestamp=2.0,
        code_version="test-version",
    )

    first = store.save_artifact(
        name="alignment",
        obj={"alignment": "cached"},
        metadata=metadata,
    )
    second = store.compute_step_hash(
        step="alignment",
        inputs={"audio": "input-hash", "transcript": "tx-hash"},
        parameters={"beam": 4},
    )

    assert first.hash == second
