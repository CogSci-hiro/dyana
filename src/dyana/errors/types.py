from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional
import traceback as _traceback


class StepStatus(str, Enum):
    """Status of a named step in a run."""
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class FailureRecord:
    """
    A structured record of a step failure or skip.

    Usage example
    -------------
        rec = FailureRecord(step_name="parse", status=StepStatus.FAILED, message="boom")
    """
    step_name: str
    status: StepStatus
    message: str
    exc_type: Optional[str] = None
    traceback: Optional[str] = None
    context: Optional[Mapping[str, Any]] = None
    caused_by: Optional[str] = None  # for SKIPPED: which dependency caused the skip

    @staticmethod
    def from_exception(*, step_name: str, exc: BaseException, context: Optional[Mapping[str, Any]]) -> "FailureRecord":
        tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        return FailureRecord(
            step_name=step_name,
            status=StepStatus.FAILED,
            message=str(exc),
            exc_type=type(exc).__name__,
            traceback=tb,
            context=context,
        )


class OptionalBackendUnavailableError(RuntimeError):
    """Raised when an optional backend was explicitly requested but is unavailable."""


class BackendConfigurationError(ValueError):
    """Raised when an optional backend is enabled but missing required configuration."""


class EvidenceTrackAlignmentError(ValueError):
    """Raised when external evidence cannot be aligned to DYANA's canonical frame grid."""
