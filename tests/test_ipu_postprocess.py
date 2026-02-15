from dyana.decode import ipu


def test_ipu_module_has_docstring() -> None:
    assert ipu.__doc__ is not None
