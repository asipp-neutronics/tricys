"""Cross-path consistency of the simulation time base (regression for silent 3600x).

TRICYS configuration time fields are authored in **hours**. ``time_base`` says how the
Modelica model is authored; when it is ``"second"`` (the default) the hour values must be
scaled by 3600 before they reach OpenModelica.

The standard engine has two execution paths:

* sequential — ``StandardSimulationEngine.execute`` -> ``ModelicaSystem.setSimulationOptions``
* concurrent — ``StandardSimulationEngine.execute`` -> ``mp_fast_job_worker`` ->
  ``run_fast_subprocess_job`` -> ``_sim_time_to_seconds``

Both paths must hand OMC the *same* stopTime/stepSize for the same configuration. Before
the fix the sequential path only scaled values that came from ``scenario.stopTime`` or from
engine overrides, so a V1-style config holding ``simulation.stop_time`` reached OMC
unscaled: 2000 (seconds) instead of 7_200_000 (seconds), i.e. 3600x shorter than ``main``
and 3600x shorter than the concurrent path — with no warning and a successful exit status.

This module is pure Python (OpenModelica is faked), so it is selected by the CI gate
(``pytest -m build_test test``, see ``.github/workflows/build_test.yml``).
"""

from __future__ import annotations

import copy
import logging

import pandas as pd
import pytest

from tricys.simulation.engines import standard as standard_mod
from tricys.simulation.engines.standard import StandardSimulationEngine
from tricys.simulation.simulation import _sim_time_to_seconds
from tricys.workflow.context import PipelineContext

pytestmark = pytest.mark.build_test

HOURS_TO_SECONDS = 3600.0
STOP_TIME_HOURS = 2000.0
STEP_SIZE_HOURS = 0.5

# V1-style configuration: the time fields live under ``simulation`` and there is no
# ``scenario`` block at all. This is exactly the shape that used to slip through unscaled.
LEGACY_V1_CONFIG = {
    "paths": {"package_path": "example_model_single/example_model.mo"},
    "simulation": {
        "model_name": "example_model.Cycle",
        "stop_time": STOP_TIME_HOURS,
        "step_size": STEP_SIZE_HOURS,
    },
}


def _execute_engine(
    monkeypatch, tmp_path, *, concurrent: bool, config: dict | None = None
) -> dict:
    """Run one job through the standard engine and capture what OMC would receive."""
    captured: dict = {}

    class _FakeSession:
        def sendExpression(self, expression):  # noqa: D401 - fake OMC session
            return "ok"

    class _FakeModelicaSystem:
        def __init__(self, *args, **kwargs):
            pass

        def setSimulationOptions(self, options):
            captured["sequential_options"] = dict(options)

        def setParameters(self, params):
            pass

        def simulate(self, resultfile=None, simargs=None):
            pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, 2.0]}).to_csv(
                resultfile, index=False
            )

    def _fake_worker(args):
        job_params, job_id, kwargs = args
        # Mirror what run_fast_subprocess_job does before invoking the executable.
        sim_config = kwargs["sim_config"]
        _sim_time_to_seconds(sim_config)
        captured.setdefault("concurrent_options", dict(sim_config))
        return job_id, job_params, "", None

    class _FakePool:
        def __init__(self, processes=None, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def imap_unordered(self, func, args):
            for item in args:
                yield func(item)

    monkeypatch.setattr(standard_mod, "get_om_session", lambda: _FakeSession())
    monkeypatch.setattr(standard_mod, "load_modelica_package", lambda *a, **k: True)
    monkeypatch.setattr(standard_mod, "ModelicaSystem", _FakeModelicaSystem)
    monkeypatch.setattr(standard_mod, "mp_fast_job_worker", _fake_worker)
    monkeypatch.setattr(standard_mod.multiprocessing, "Pool", _FakePool)

    temp_dir = tmp_path / ("concurrent" if concurrent else "sequential")
    temp_dir.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(config if config is not None else LEGACY_V1_CONFIG)
    config["paths"]["temp_dir"] = str(temp_dir)

    context = PipelineContext(
        config=config,
        base_dir=str(tmp_path),
        temp_dir=str(temp_dir),
        results_dir=str(tmp_path / "results"),
        package_path=str(tmp_path / "example_model.mo"),
        model_name="example_model.Cycle",
        jobs=[{}],
        compiled_artifacts={
            "exe": str(tmp_path / "model"),
            "xml": str(tmp_path / "model_init.xml"),
            "om_bin": "/usr/bin",
        },
    )

    StandardSimulationEngine(parallel=concurrent).execute(context)
    return captured


def test_sequential_path_scales_hour_authored_config(monkeypatch, tmp_path):
    """A V1-style config must reach OMC in the model's base unit (seconds)."""
    captured = _execute_engine(monkeypatch, tmp_path, concurrent=False)

    options = captured["sequential_options"]
    assert options["stopTime"] == pytest.approx(STOP_TIME_HOURS * HOURS_TO_SECONDS)
    assert options["stepSize"] == pytest.approx(STEP_SIZE_HOURS * HOURS_TO_SECONDS)


def test_concurrent_path_scales_hour_authored_config(monkeypatch, tmp_path):
    """The concurrent path is the reference behaviour: it must keep scaling."""
    captured = _execute_engine(monkeypatch, tmp_path, concurrent=True)

    options = captured["concurrent_options"]
    assert options["stop_time"] == pytest.approx(STOP_TIME_HOURS * HOURS_TO_SECONDS)
    assert options["step_size"] == pytest.approx(STEP_SIZE_HOURS * HOURS_TO_SECONDS)


def test_paths_agree_on_the_same_configuration(monkeypatch, tmp_path):
    """Same config, either execution path -> identical time values for OMC."""
    sequential = _execute_engine(monkeypatch, tmp_path, concurrent=False)
    concurrent = _execute_engine(monkeypatch, tmp_path, concurrent=True)

    sequential_options = sequential["sequential_options"]
    concurrent_options = concurrent["concurrent_options"]

    assert sequential_options["stopTime"] == pytest.approx(
        concurrent_options["stop_time"]
    )
    assert sequential_options["stepSize"] == pytest.approx(
        concurrent_options["step_size"]
    )


# V2-style configuration: time fields live under ``scenario`` and the converter in
# ``config_utils`` may already have scaled them (setting ``_time_unit``). Either way the
# engine must scale exactly once.
V2_SCENARIO_CONFIG = {
    "paths": {"package_path": "example_model_single/example_model.mo"},
    "scenario": {
        "model_name": "example_model.Cycle",
        "stopTime": 10.0,
        "stepSize": 0.1,
        "time_base": "second",
    },
}


def test_v2_scenario_time_fields_are_converted_exactly_once(monkeypatch, tmp_path):
    """Guards the opposite failure mode: scaling twice would give 1.296e8 s."""
    captured = _execute_engine(
        monkeypatch, tmp_path, concurrent=False, config=V2_SCENARIO_CONFIG
    )

    options = captured["sequential_options"]
    assert options["stopTime"] == pytest.approx(10.0 * HOURS_TO_SECONDS)
    assert options["stepSize"] == pytest.approx(0.1 * HOURS_TO_SECONDS)


# ------------------------------------------------------- legacy-shape warning (AC2)


def test_legacy_config_without_time_base_warns(monkeypatch, tmp_path, caplog):
    """A 1.0-era config must not silently inherit the SI default."""
    with caplog.at_level(
        logging.WARNING, logger="tricys.simulation.engines.standard"
    ):
        _execute_engine(monkeypatch, tmp_path, concurrent=False)

    messages = [r.getMessage() for r in caplog.records]
    assert any("time_base" in m and "3600" in m for m in messages), messages


def test_v2_config_without_time_base_does_not_warn(monkeypatch, tmp_path, caplog):
    """A ``scenario`` block is the 2.0 shape: it may legitimately rely on the default."""
    with caplog.at_level(
        logging.WARNING, logger="tricys.simulation.engines.standard"
    ):
        _execute_engine(
            monkeypatch, tmp_path, concurrent=False, config=V2_SCENARIO_CONFIG
        )

    messages = [r.getMessage() for r in caplog.records]
    assert not any("time_base" in m and "3600" in m for m in messages), messages


def test_explicit_time_base_does_not_warn(monkeypatch, tmp_path, caplog):
    """An explicit declaration is the answer, so there is nothing to warn about."""
    config = copy.deepcopy(LEGACY_V1_CONFIG)
    config["simulation"]["time_base"] = "hour"

    with caplog.at_level(
        logging.WARNING, logger="tricys.simulation.engines.standard"
    ):
        _execute_engine(monkeypatch, tmp_path, concurrent=False, config=config)

    messages = [r.getMessage() for r in caplog.records]
    assert not any("time_base" in m and "3600" in m for m in messages), messages
