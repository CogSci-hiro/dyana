import re

from dyana import version


def test_version_string_is_semverish() -> None:
    assert isinstance(version.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-][0-9A-Za-z.]+)?", version.__version__) is not None

