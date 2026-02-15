import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "dyana.io.artifacts",
        "dyana.io.audio",
        "dyana.io.bids_like",
        "dyana.io.praat_textgrid",
    ],
)
def test_io_modules_have_docstrings(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__ is not None

