"""Registered placeholder steps for the annotation pipeline."""

from __future__ import annotations

from collections.abc import Mapping

from .registry import STEP_REGISTRY, register_step
from .types import PipelineStep


def _run_transcription_step(inputs: Mapping[str, object]) -> dict[str, object]:
    """
    Produce a transcript bundle from audio.

    Parameters
    ----------
    inputs
        Mapping containing the ``audio`` contract object.
    """

    del inputs
    raise NotImplementedError("Transcription step not implemented yet")


def _run_alignment_step(inputs: Mapping[str, object]) -> dict[str, object]:
    """
    Produce an alignment bundle from audio and transcript inputs.

    Parameters
    ----------
    inputs
        Mapping containing the ``audio`` and ``transcript`` contract objects.
    """

    del inputs
    raise NotImplementedError("Alignment step not implemented yet")


def _run_evidence_step(inputs: Mapping[str, object]) -> dict[str, object]:
    """
    Produce an evidence bundle from audio.

    Parameters
    ----------
    inputs
        Mapping containing the ``audio`` contract object.
    """

    del inputs
    raise NotImplementedError("Evidence step not implemented yet")


def _run_decode_step(inputs: Mapping[str, object]) -> dict[str, object]:
    """
    Produce a decode result from alignment and evidence inputs.

    Parameters
    ----------
    inputs
        Mapping containing the ``alignment`` and ``evidence`` contract objects.
    """

    del inputs
    raise NotImplementedError("Decode step not implemented yet")


def _run_diagnostics_step(inputs: Mapping[str, object]) -> dict[str, object]:
    """
    Produce diagnostics from decode outputs.

    Parameters
    ----------
    inputs
        Mapping containing the ``decode`` contract object.
    """

    del inputs
    raise NotImplementedError("Diagnostics step not implemented yet")


_DEFAULT_STEPS = (
    PipelineStep(
        name="transcription",
        inputs=("audio",),
        outputs=("transcript",),
        run=_run_transcription_step,
    ),
    PipelineStep(
        name="alignment",
        inputs=("audio", "transcript"),
        outputs=("alignment",),
        run=_run_alignment_step,
    ),
    PipelineStep(
        name="evidence",
        inputs=("audio",),
        outputs=("evidence",),
        run=_run_evidence_step,
    ),
    PipelineStep(
        name="decode",
        inputs=("alignment", "evidence"),
        outputs=("decode",),
        run=_run_decode_step,
    ),
    PipelineStep(
        name="diagnostics",
        inputs=("decode",),
        outputs=("diagnostics",),
        run=_run_diagnostics_step,
    ),
)


for _step in _DEFAULT_STEPS:
    if _step.name not in STEP_REGISTRY:
        register_step(_step)
