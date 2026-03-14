from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dyana.artifacts.store import ArtifactStore
from dyana.pipeline.registry import STEP_REGISTRY, _OUTPUT_REGISTRY, register_step
from dyana.pipeline.runner import run_pipeline, summarize_run
from dyana.pipeline.types import PipelineStep


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    step_snapshot = dict(STEP_REGISTRY)
    output_snapshot = dict(_OUTPUT_REGISTRY)
    STEP_REGISTRY.clear()
    _OUTPUT_REGISTRY.clear()
    monkeypatch.setenv("DYANA_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    try:
        yield
    finally:
        STEP_REGISTRY.clear()
        STEP_REGISTRY.update(step_snapshot)
        _OUTPUT_REGISTRY.clear()
        _OUTPUT_REGISTRY.update(output_snapshot)


def test_runner_resolves_dependencies_in_order() -> None:
    execution_order: list[str] = []

    def run_transcript(inputs: dict[str, object]) -> dict[str, object]:
        execution_order.append("transcription")
        return {"transcript": f"tx:{inputs['audio']}"}

    def run_alignment(inputs: dict[str, object]) -> dict[str, object]:
        execution_order.append("alignment")
        return {"alignment": f"al:{inputs['transcript']}"}

    def run_evidence(inputs: dict[str, object]) -> dict[str, object]:
        execution_order.append("evidence")
        return {"evidence": f"ev:{inputs['audio']}"}

    def run_decode(inputs: dict[str, object]) -> dict[str, object]:
        execution_order.append("decode")
        return {"decode": f"de:{inputs['alignment']}|{inputs['evidence']}"}

    def run_diagnostics(inputs: dict[str, object]) -> dict[str, object]:
        execution_order.append("diagnostics")
        return {"diagnostics": f"diag:{inputs['decode']}"}

    register_step(PipelineStep("transcription", ("audio",), ("transcript",), run_transcript))
    register_step(PipelineStep("alignment", ("audio", "transcript"), ("alignment",), run_alignment))
    register_step(PipelineStep("evidence", ("audio",), ("evidence",), run_evidence))
    register_step(PipelineStep("decode", ("alignment", "evidence"), ("decode",), run_decode))
    register_step(PipelineStep("diagnostics", ("decode",), ("diagnostics",), run_diagnostics))

    state = run_pipeline(initial_state={"audio": "clip.wav"}, target_outputs=("diagnostics",))

    assert execution_order == ["transcription", "alignment", "evidence", "decode", "diagnostics"]
    assert state["diagnostics"].startswith("diag:")


def test_runner_uses_cached_initial_outputs() -> None:
    execution_order: list[str] = []

    def run_alignment(inputs: dict[str, object]) -> dict[str, object]:
        execution_order.append("alignment")
        return {"alignment": f"al:{inputs['transcript']}"}

    register_step(PipelineStep("alignment", ("audio", "transcript"), ("alignment",), run_alignment))

    state = run_pipeline(
        initial_state={"audio": "clip.wav", "transcript": "cached transcript"},
        target_outputs=("alignment",),
    )

    assert execution_order == ["alignment"]
    assert state["alignment"] == "al:cached transcript"


def test_runner_raises_for_unknown_required_output() -> None:
    with pytest.raises(ValueError, match="No pipeline step produces required output 'decode'"):
        run_pipeline(initial_state={"audio": "clip.wav"}, target_outputs=("decode",))


def test_runner_raises_for_missing_inputs_before_execution() -> None:
    register_step(
        PipelineStep(
            name="alignment",
            inputs=("audio", "transcript"),
            outputs=("alignment",),
            run=lambda inputs: {"alignment": inputs["audio"]},
        )
    )

    with pytest.raises(ValueError, match="No pipeline step produces required output 'transcript'"):
        run_pipeline(initial_state={"audio": "clip.wav"}, target_outputs=("alignment",))


def test_runner_detects_circular_dependencies() -> None:
    register_step(PipelineStep("make_alignment", ("decode",), ("alignment",), lambda _: {"alignment": object()}))
    register_step(PipelineStep("make_decode", ("alignment",), ("decode",), lambda _: {"decode": object()}))

    with pytest.raises(ValueError, match="Circular dependency detected"):
        run_pipeline(initial_state={}, target_outputs=("decode",))


def test_runner_raises_when_step_does_not_produce_declared_output() -> None:
    register_step(PipelineStep("transcription", ("audio",), ("transcript",), lambda _: {}))

    with pytest.raises(ValueError, match="did not produce declared outputs: transcript"):
        run_pipeline(initial_state={"audio": "clip.wav"}, target_outputs=("transcript",))


def test_runner_rejects_non_mapping_step_results() -> None:
    register_step(PipelineStep("transcription", ("audio",), ("transcript",), lambda _: "bad-result"))

    with pytest.raises(TypeError, match="must return a mapping of outputs"):
        run_pipeline(initial_state={"audio": "clip.wav"}, target_outputs=("transcript",))


def test_runner_reuses_cached_artifact_for_same_step_hash(tmp_path: Path) -> None:
    artifact_store = ArtifactStore(root=tmp_path / "artifacts")
    call_count = 0

    def run_transcript(inputs: dict[str, object]) -> dict[str, object]:
        nonlocal call_count
        call_count += 1
        return {"transcript": f"tx:{inputs['audio']}"}

    register_step(
        PipelineStep(
            "transcription",
            ("audio",),
            ("transcript",),
            run_transcript,
            parameters={"model": "baseline"},
        )
    )

    first_state = run_pipeline(
        initial_state={"audio": "clip.wav"},
        target_outputs=("transcript",),
        artifact_store=artifact_store,
        run_id="run-1",
    )
    second_state = run_pipeline(
        initial_state={"audio": "clip.wav"},
        target_outputs=("transcript",),
        artifact_store=artifact_store,
        run_id="run-2",
    )

    assert call_count == 1
    assert first_state["transcript"] == second_state["transcript"]

    summary = summarize_run("run-2", artifact_root=artifact_store.root)
    assert summary["steps_executed"] == ["transcription"]
    assert summary["failures"] == []
