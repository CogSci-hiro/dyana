from __future__ import annotations

from collections.abc import Iterator

import pytest

from dyana.pipeline.registry import STEP_REGISTRY, _OUTPUT_REGISTRY, register_step
from dyana.pipeline.types import PipelineStep


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    step_snapshot = dict(STEP_REGISTRY)
    output_snapshot = dict(_OUTPUT_REGISTRY)
    STEP_REGISTRY.clear()
    _OUTPUT_REGISTRY.clear()
    try:
        yield
    finally:
        STEP_REGISTRY.clear()
        STEP_REGISTRY.update(step_snapshot)
        _OUTPUT_REGISTRY.clear()
        _OUTPUT_REGISTRY.update(output_snapshot)


def test_register_step_stores_step() -> None:
    step = PipelineStep(name="transcription", inputs=("audio",), outputs=("transcript",), run=lambda _: {})

    register_step(step)

    assert STEP_REGISTRY["transcription"] is step


def test_register_step_rejects_duplicate_name() -> None:
    first = PipelineStep(name="transcription", inputs=("audio",), outputs=("transcript",), run=lambda _: {})
    second = PipelineStep(name="transcription", inputs=("audio",), outputs=("alignment",), run=lambda _: {})
    register_step(first)

    with pytest.raises(ValueError, match="already registered"):
        register_step(second)


def test_register_step_rejects_output_collisions() -> None:
    first = PipelineStep(name="transcription", inputs=("audio",), outputs=("transcript",), run=lambda _: {})
    second = PipelineStep(name="translation", inputs=("audio",), outputs=("transcript",), run=lambda _: {})
    register_step(first)

    with pytest.raises(ValueError, match="already produced"):
        register_step(second)
