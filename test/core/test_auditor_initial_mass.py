"""The online mass auditor must validate its initial baseline.

Before this contract, ``_finalize_initialization`` summed whatever the FMUs returned and
stored it as ``initial_mass``: a NaN, a negative inventory, or a variable set that could
not be read at all produced a plausible-looking baseline, and every later mass balance was
computed against it. Silent wrong numbers are the worst failure mode for an auditor, so
initialization now refuses a baseline it cannot trust.

Pure Python (the OMS runtime is faked), so it is selected by the CI gate
(``pytest -m build_test test``).
"""

from __future__ import annotations

import logging

import pytest

from tricys.plugins.auditor.offline import AuditorConfig
from tricys.plugins.auditor.online import (
    AuditorInitializationError,
    OnlineGlobalAuditor,
)

pytestmark = pytest.mark.build_test


class _FakeRuntime:
    def __init__(self, values: dict[tuple[str, str], object]):
        self.values = values

    def get_value(self, system_name, comp_name, var_name):
        key = (comp_name, var_name)
        if key not in self.values:
            raise KeyError(f"{comp_name}.{var_name} not readable")
        return self.values[key]


def _auditor(bindings: list[tuple[str, str]]) -> OnlineGlobalAuditor:
    auditor = OnlineGlobalAuditor(AuditorConfig(enabled=True))
    auditor.inv_bindings = list(bindings)
    return auditor


def test_valid_inventories_establish_the_baseline():
    runtime = _FakeRuntime({("a", "I[1]"): 1.5, ("b", "I[1]"): 2.5})
    auditor = _auditor([("a", "I[1]"), ("b", "I[1]")])

    auditor._finalize_initialization(runtime, "default", 1.0)

    assert auditor.state.initial_mass == pytest.approx(5.0)
    assert auditor.state.current_mass == pytest.approx(5.0)
    assert auditor._initialized is True


def test_non_finite_inventory_is_rejected():
    runtime = _FakeRuntime({("a", "I[1]"): float("nan")})
    auditor = _auditor([("a", "I[1]")])

    with pytest.raises(AuditorInitializationError, match="not finite"):
        auditor._finalize_initialization(runtime, "default", 0.0)

    assert auditor._initialized is False


def test_infinite_inventory_is_rejected():
    runtime = _FakeRuntime({("a", "I[1]"): float("inf")})
    auditor = _auditor([("a", "I[1]")])

    with pytest.raises(AuditorInitializationError, match="not finite"):
        auditor._finalize_initialization(runtime, "default", 0.0)


def test_negative_inventory_is_rejected():
    runtime = _FakeRuntime({("a", "I[1]"): 3.0, ("b", "I[1]"): -0.5})
    auditor = _auditor([("a", "I[1]"), ("b", "I[1]")])

    with pytest.raises(AuditorInitializationError, match="negative"):
        auditor._finalize_initialization(runtime, "default", 0.0)


def test_unreadable_inventory_set_is_rejected():
    runtime = _FakeRuntime({})  # every lookup raises
    auditor = _auditor([("a", "I[1]"), ("b", "I[1]")])

    with pytest.raises(AuditorInitializationError, match="could be read"):
        auditor._finalize_initialization(runtime, "default", 0.0)


def test_partially_readable_inventory_set_is_accepted():
    """Losing one variable is a warning; losing all of them is not a baseline."""
    runtime = _FakeRuntime({("a", "I[1]"): 4.0})
    auditor = _auditor([("a", "I[1]"), ("missing", "I[1]")])

    auditor._finalize_initialization(runtime, "default", 0.0)

    assert auditor.state.initial_mass == pytest.approx(4.0)


def test_no_matching_inventory_pattern_is_rejected():
    auditor = _auditor([])

    with pytest.raises(AuditorInitializationError, match="no inventory variables matched"):
        auditor._finalize_initialization(_FakeRuntime({}), "default", 0.0)


def test_invalid_processor_inventory_is_rejected():
    runtime = _FakeRuntime({("a", "I[1]"): 1.0})
    auditor = _auditor([("a", "I[1]")])

    with pytest.raises(AuditorInitializationError, match="processor inventory"):
        auditor._finalize_initialization(runtime, "default", -2.0)

    auditor = _auditor([("a", "I[1]")])
    with pytest.raises(AuditorInitializationError, match="processor inventory"):
        auditor._finalize_initialization(runtime, "default", float("nan"))


def test_zero_inventory_is_accepted_with_a_warning(caplog):
    runtime = _FakeRuntime({("a", "I[1]"): 0.0})
    auditor = _auditor([("a", "I[1]")])

    with caplog.at_level(logging.WARNING, logger="tricys.plugins.auditor.online"):
        auditor._finalize_initialization(runtime, "default", 0.0)

    assert auditor.state.initial_mass == 0.0
    assert any("exactly 0" in r.getMessage() for r in caplog.records)
