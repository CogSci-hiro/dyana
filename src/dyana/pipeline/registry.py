"""Registry for declarative pipeline steps."""

from __future__ import annotations

from .types import PipelineStep

STEP_REGISTRY: dict[str, PipelineStep] = {}
_OUTPUT_REGISTRY: dict[str, str] = {}


def register_step(step: PipelineStep) -> None:
    """
    Register a pipeline step and validate name/output uniqueness.

    Parameters
    ----------
    step
        Step definition to register.
    """

    if step.name in STEP_REGISTRY:
        raise ValueError(f"Step '{step.name}' is already registered.")

    for output_name in step.outputs:
        if output_name in _OUTPUT_REGISTRY:
            producer_name = _OUTPUT_REGISTRY[output_name]
            raise ValueError(
                f"Output '{output_name}' is already produced by step '{producer_name}'."
            )

    STEP_REGISTRY[step.name] = step
    for output_name in step.outputs:
        _OUTPUT_REGISTRY[output_name] = step.name


def get_step_for_output(output_name: str) -> PipelineStep | None:
    """Return the registered step that produces ``output_name``."""

    step_name = _OUTPUT_REGISTRY.get(output_name)
    if step_name is None:
        return None
    return STEP_REGISTRY[step_name]
