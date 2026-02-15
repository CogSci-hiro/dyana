from dyana.decode import state_space


def test_state_space_module_has_docstring() -> None:
    assert state_space.__doc__ is not None
