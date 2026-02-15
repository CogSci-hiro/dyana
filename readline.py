"""
Local stub for `readline`.

In some environments (notably certain macOS/conda builds), importing the native
`readline` extension can segfault. Pytest imports `readline` unconditionally
very early during startup (see `_pytest.capture._readline_workaround`), which
would crash the entire test run before collection.

This pure-Python stub intentionally provides no API; pytest only needs the
import side-effect. If you need real readline functionality in production code,
import it from a known-good environment instead of relying on this stub.
"""

