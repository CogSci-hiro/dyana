from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dyana.core.contracts.diagnostics import DiagnosticsBundle


def test_diagnostics_contract_dataclass_can_be_constructed() -> None:
    bundle = DiagnosticsBundle(metrics={"coverage": 1.0}, flags=["ok"])

    assert bundle.metrics["coverage"] == 1.0
    assert bundle.flags == ["ok"]


def test_diagnostics_contract_dataclass_is_frozen() -> None:
    bundle = DiagnosticsBundle(metrics={"coverage": 1.0})

    with pytest.raises(FrozenInstanceError):
        bundle.flags = ["warning"]  # type: ignore[misc]
