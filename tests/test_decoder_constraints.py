from dyana.decode import constraints


def test_constraints_module_has_docstring() -> None:
    assert constraints.__doc__ is not None
