from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Optional, TypeVar

from .reporter import ErrorReporter

T = TypeVar("T")


@contextmanager
def step(
    step_name: str,
    reporter: ErrorReporter,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> Iterator[None]:
    """
    Context manager wrapping a named step.

    Behavior
    --------
    - debug mode: exception is re-raised (hard stop).
    - run mode: exception is recorded and suppressed; caller continues after the block.

    Usage example
    -------------
        with step("load_audio", reporter, context={"subject": "sub-001"}):
            audio = load_audio(...)
    """
    try:
        yield
    except BaseException as exc:
        reporter.mark_failed(step_name=step_name, exc=exc, context=context)
        if reporter.cfg.mode == "debug":
            raise
    else:
        reporter.mark_ok(step_name)


def guard(
    step_name: str,
    reporter: ErrorReporter,
    fn: Callable[[], T],
    *,
    context: Optional[Mapping[str, Any]] = None,
    default: Optional[T] = None,
) -> Optional[T]:
    """
    Execute a callable under error-handling policy.

    Returns
    -------
    value
        The callable result on success; otherwise `default`.

    Usage example
    -------------
        audio = guard("load_audio", reporter, lambda: load_audio(path), default=None)
        if audio is None:
            return
    """
    try:
        result = fn()
    except BaseException as exc:
        reporter.mark_failed(step_name=step_name, exc=exc, context=context)
        if reporter.cfg.mode == "debug":
            raise
        return default
    else:
        reporter.mark_ok(step_name)
        return result

