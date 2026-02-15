import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "dyana.core.cache",
        "dyana.core.calibrate",
        "dyana.core.types",
    ],
)
def test_core_modules_have_docstrings(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__ is not None
