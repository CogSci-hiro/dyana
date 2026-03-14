from __future__ import annotations

from dyana.pipeline.registry import STEP_REGISTRY
from dyana.pipeline.steps import _DEFAULT_STEPS


def test_default_steps_are_registered() -> None:
    expected_names = {step.name for step in _DEFAULT_STEPS}

    assert expected_names.issubset(STEP_REGISTRY)
    assert STEP_REGISTRY["transcription"].inputs == ("audio",)
    assert STEP_REGISTRY["alignment"].outputs == ("alignment",)
    assert STEP_REGISTRY["decode"].inputs == ("alignment", "evidence")
