import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "dyana.evidence.base",
        "dyana.evidence.bundle",
        "dyana.evidence.diarization",
        "dyana.evidence.linguistic_hints",
        "dyana.evidence.mic_priors",
        "dyana.evidence.prosody",
        "dyana.evidence.separation",
        "dyana.evidence.vad",
    ],
)
def test_evidence_modules_have_docstrings(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__ is not None
