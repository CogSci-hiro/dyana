"""Public orchestration entrypoint for the dependency-safe pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from dyana.api.types import AnnotationResult
from dyana.artifacts.paths import get_run_dir
from dyana.core.contracts.audio import AudioInput
from dyana.io.artifacts import write_run_artifacts
from dyana.pipeline.runner import summarize_run

from .assemble import assemble_annotation
from .runner import run_pipeline


def run_annotation_pipeline(audio: AudioInput) -> AnnotationResult:
    """
    Execute the annotation pipeline from a normalized audio contract.

    Parameters
    ----------
    audio
        Audio contract object supplied by the public API layer.
    """

    from . import steps as _steps  # noqa: F401

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + f"-{uuid4().hex[:6]}"
    state = run_pipeline(
        initial_state={"audio": audio},
        target_outputs=("transcript", "alignment", "decode", "diagnostics"),
        run_id=run_id,
    )
    result = assemble_annotation(
        transcript=state["transcript"],
        alignment=state["alignment"],
        decode=state["decode"],
        diagnostics=state["diagnostics"],
    )
    run_summary = summarize_run(run_id)
    write_run_artifacts(
        annotation_result=result,
        run_dir=get_run_dir(run_id),
        audio_path=str(audio.path) if audio.path is not None else None,
        duration=audio.duration,
        pipeline_steps=run_summary["steps_executed"],
        evidence_bundle=state.get("evidence"),
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return result
