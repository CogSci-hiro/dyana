from dyana.decode import decoder


def test_decoder_module_has_docstring() -> None:
    assert decoder.__doc__ is not None

