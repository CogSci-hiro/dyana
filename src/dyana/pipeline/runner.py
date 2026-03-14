"""Dependency-aware pipeline runner."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from dyana.artifacts.metadata import ArtifactMetadata
from dyana.artifacts.paths import get_artifact_root
from dyana.artifacts.store import ArtifactStore
from dyana.version import __version__

from .registry import get_step_for_output
from .types import PipelineStep


def _resolve_required_steps(
    target_name: str,
    available_keys: set[str],
    ordered_steps: list[PipelineStep],
    scheduled_steps: set[str],
    visiting_outputs: set[str],
) -> None:
    if target_name in available_keys:
        return

    if target_name in visiting_outputs:
        raise ValueError(f"Circular dependency detected while resolving '{target_name}'.")

    step = get_step_for_output(target_name)
    if step is None:
        raise ValueError(f"No pipeline step produces required output '{target_name}'.")

    visiting_outputs.add(target_name)
    for input_name in step.inputs:
        _resolve_required_steps(
            target_name=input_name,
            available_keys=available_keys,
            ordered_steps=ordered_steps,
            scheduled_steps=scheduled_steps,
            visiting_outputs=visiting_outputs,
        )
    visiting_outputs.remove(target_name)

    if step.name not in scheduled_steps:
        ordered_steps.append(step)
        scheduled_steps.add(step.name)
        available_keys.update(step.outputs)


def _select_steps(
    initial_state: dict[str, object],
    target_outputs: tuple[str, ...],
) -> list[PipelineStep]:
    available_keys = set(initial_state)
    ordered_steps: list[PipelineStep] = []
    scheduled_steps: set[str] = set()

    for output_name in target_outputs:
        _resolve_required_steps(
            target_name=output_name,
            available_keys=available_keys,
            ordered_steps=ordered_steps,
            scheduled_steps=scheduled_steps,
            visiting_outputs=set(),
        )

    return ordered_steps


def run_pipeline(
    initial_state: dict[str, object],
    target_outputs: tuple[str, ...],
    *,
    artifact_store: ArtifactStore | None = None,
    run_id: str | None = None,
) -> dict[str, object]:
    """
    Resolve and execute the steps required to compute target outputs.

    Parameters
    ----------
    initial_state
        Pre-populated state objects available to the pipeline.
    target_outputs
        State keys that must be available after execution.

    Returns
    -------
    dict[str, object]
        Final state including all cached intermediate outputs.
    """

    store = artifact_store or ArtifactStore()
    resolved_run_id = run_id or uuid4().hex
    state = dict(initial_state)
    steps_to_run = _select_steps(initial_state=state, target_outputs=target_outputs)
    input_hashes = {name: store.hash_object(value) for name, value in state.items()}
    step_logs: list[dict[str, object]] = []
    produced_artifacts: dict[str, str] = {}

    try:
        for step in steps_to_run:
            missing_inputs = tuple(name for name in step.inputs if name not in state)
            if missing_inputs:
                missing_list = ", ".join(missing_inputs)
                raise ValueError(f"Step '{step.name}' is missing required inputs: {missing_list}.")

            step_inputs = {name: state[name] for name in step.inputs}
            dependency_hashes = {name: input_hashes[name] for name in step.inputs}
            step_parameters = dict(step.parameters)
            step_hash = store.compute_step_hash(
                step=step.name,
                inputs=dependency_hashes,
                parameters=step_parameters,
            )
            cache_hit = store.artifact_exists(step_hash)

            log_entry = {
                "artifact_hash": step_hash,
                "end_time": None,
                "inputs": dependency_hashes,
                "outputs": list(step.outputs),
                "parameters": step_parameters,
                "start_time": time.time(),
                "status": "cached" if cache_hit else "running",
                "step": step.name,
            }

            try:
                if cache_hit:
                    step_result = store.load_artifact(step_hash)
                else:
                    step_result = step.run(step_inputs)

                if not isinstance(step_result, Mapping):
                    raise TypeError(
                        f"Step '{step.name}' must return a mapping of outputs, got {type(step_result).__name__}."
                    )

                if cache_hit:
                    artifact_hash = step_hash
                else:
                    metadata = ArtifactMetadata(
                        step=step.name,
                        inputs=dependency_hashes,
                        parameters=step_parameters,
                        timestamp=time.time(),
                        code_version=__version__,
                    )
                    artifact = store.save_artifact(step.name, dict(step_result), metadata)
                    artifact_hash = artifact.hash

                state.update(step_result)

                missing_outputs = tuple(name for name in step.outputs if name not in state)
                if missing_outputs:
                    missing_list = ", ".join(missing_outputs)
                    raise ValueError(f"Step '{step.name}' did not produce declared outputs: {missing_list}.")

                for output_name in step.outputs:
                    input_hashes[output_name] = artifact_hash
                    produced_artifacts[output_name] = artifact_hash

                log_entry["artifact_hash"] = artifact_hash
                log_entry["status"] = "cached" if cache_hit else "executed"
            except Exception as exc:
                log_entry["status"] = "failed"
                log_entry["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                log_entry["end_time"] = time.time()
                step_logs.append(log_entry)

        final_missing = tuple(name for name in target_outputs if name not in state)
        if final_missing:
            missing_list = ", ".join(final_missing)
            raise ValueError(f"Pipeline did not produce required outputs: {missing_list}.")

        return state
    finally:
        _write_run_records(
            store=store,
            run_id=resolved_run_id,
            target_outputs=target_outputs,
            step_logs=step_logs,
            produced_artifacts=produced_artifacts,
        )


def summarize_run(run_id: str, *, artifact_root: Path | None = None) -> dict[str, object]:
    root = artifact_root or get_artifact_root()
    run_dir = root / "runs" / run_id
    summary_path = run_dir / "pipeline_summary.json"
    step_logs_path = run_dir / "step_logs.json"
    summary = json.loads(summary_path.read_text())
    step_logs = json.loads(step_logs_path.read_text())
    failures = [entry for entry in step_logs if entry["status"] == "failed"]
    return {
        "artifacts_produced": summary["artifacts_produced"],
        "failures": failures,
        "run_id": summary["run_id"],
        "steps_executed": [entry["step"] for entry in step_logs],
    }


def _write_run_records(
    *,
    store: ArtifactStore,
    run_id: str,
    target_outputs: tuple[str, ...],
    step_logs: list[dict[str, object]],
    produced_artifacts: dict[str, str],
) -> None:
    run_dir = store.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "artifacts_produced": produced_artifacts,
        "run_id": run_id,
        "status": "failed" if any(entry["status"] == "failed" for entry in step_logs) else "ok",
        "step_count": len(step_logs),
        "target_outputs": list(target_outputs),
    }
    (run_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (run_dir / "step_logs.json").write_text(json.dumps(step_logs, indent=2, sort_keys=True))
