"""
errors subpackage: robust error handling + logging for long-running workflows.

Key primitives
--------------
- ErrorHandlingConfig: global config (mode, log paths, JSONL, etc.)
- configure_logging(): console + file logging, optional JSONL event logger
- ErrorReporter: captures failures/skips and renders end-of-run report
- step(): context manager to wrap a named pipeline step
- guard(): one-liner wrapper for callables
- Pipeline: dependency-aware runner that skips meaningless downstream steps
"""

from .config import ErrorHandlingConfig
from .logging import configure_logging, JsonlEventLogger
from .reporter import ErrorReporter
from .guards import step, guard
from .pipeline import Pipeline

__all__ = [
    "ErrorHandlingConfig",
    "JsonlEventLogger",
    "configure_logging",
    "ErrorReporter",
    "step",
    "guard",
    "Pipeline",
]
