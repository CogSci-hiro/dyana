from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, TypeVar

from .reporter import ErrorReporter
from .guards import step

T = TypeVar("T")


@dataclass(frozen=True)
class _StepDef:
    name: str
    fn: Callable[[], Any]
    deps: tuple[str, ...]
    context: Optional[Mapping[str, Any]]


class Pipeline:
    """
    Dependency-aware step runner.

    Rules
    -----
    - A step runs only if all dependencies are OK.
    - If any dependency FAILED or SKIPPED, this step is SKIPPED.
    - In run mode, pipeline continues with independent steps.
    - In debug mode, first failure raises.

    Usage example
    -------------
        pipe = Pipeline(reporter)

        pipe.add("load_audio", lambda: load_audio(path), context={"subject": "sub-001"})
        pipe.add("extract_features", lambda: extract(audio), deps=["load_audio"])
        pipe.add("fit", lambda: fit_model(feats), deps=["extract_features"])

        results = pipe.run()
        reporter.print_summary()
    """

    def __init__(self, reporter: ErrorReporter) -> None:
        self._reporter = reporter
        self._steps: Dict[str, _StepDef] = {}

    def add(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        deps: Optional[List[str]] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Register a named step with optional dependencies."""
        if name in self._steps:
            raise ValueError(f"Duplicate step name: {name}")
        self._steps[name] = _StepDef(name=name, fn=fn, deps=tuple(deps or []), context=context)

    def run(self) -> Dict[str, Any]:
        """
        Execute the pipeline in dependency order.

        Returns
        -------
        results
            Mapping from step name to return value (only for steps that ran successfully).
            Skipped/failed steps will not be present.

        Notes
        -----
        If cfg.max_failures is set, pipeline stops scheduling new work once reached.
        """
        results: Dict[str, Any] = {}
        remaining: Set[str] = set(self._steps.keys())
        executed_or_decided: Set[str] = set()

        def _max_failures_reached() -> bool:
            if self._reporter.cfg.mode != "run":
                return False
            mf = self._reporter.cfg.max_failures
            return mf is not None and self._reporter.failures_count() >= mf

        def _skip_all_remaining(*, caused_by: str) -> None:
            for n in sorted(remaining):
                if self._reporter.status(n) is None:
                    self._reporter.mark_skipped(step_name=n, caused_by=caused_by, context=self._steps[n].context)

        while remaining:
            progressed = False

            # If threshold already reached before entering this pass, skip everything left.
            if _max_failures_reached():
                _skip_all_remaining(caused_by="max_failures")
                break

            for name in sorted(list(remaining)):
                sdef = self._steps[name]

                # If deps not yet decided, can't run/skip yet
                if any(dep not in executed_or_decided for dep in sdef.deps):
                    continue

                # If any dep failed/skipped -> skip
                bad_dep = next((dep for dep in sdef.deps if not self._reporter.ok(dep)), None)
                if bad_dep is not None:
                    self._reporter.mark_skipped(step_name=name, caused_by=bad_dep, context=sdef.context)
                    remaining.remove(name)
                    executed_or_decided.add(name)
                    progressed = True

                    if _max_failures_reached():
                        _skip_all_remaining(caused_by="max_failures")
                        remaining.clear()
                    continue

                # All deps OK -> run
                with step(name, self._reporter, context=sdef.context):
                    out = sdef.fn()
                    results[name] = out

                remaining.remove(name)
                executed_or_decided.add(name)
                progressed = True

                # IMPORTANT: enforce max_failures immediately after a step completes
                if _max_failures_reached():
                    _skip_all_remaining(caused_by="max_failures")
                    remaining.clear()
                    break

            if not progressed and remaining:
                unresolved = ", ".join(sorted(remaining))
                raise RuntimeError(
                    "Pipeline could not make progress (cycle or undefined deps). "
                    f"Remaining steps: {unresolved}"
                )

        return results
