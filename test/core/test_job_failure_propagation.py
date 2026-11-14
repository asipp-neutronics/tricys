"""Any failed simulation job must fail the run (non-zero exit code).

Before this contract existed, a diverged solver or a crashed job was only written to the
log: ``StandardSimulationEngine`` reported "Completed N/M successfully" and the process
exited 0, so a sweep with silently missing rows looked like a successful run.

Policy: **any** job failure fails the whole invocation, on both the sequential and the
concurrent path, and for co-simulation (offline table) as well.

Pure Python (OpenModelica is faked), so it is selected by the CI gate
(``pytest -m build_test test``, see ``.github/workflows/build_test.yml``).
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from tricys.simulation import simulation as simulation_mod
from tricys.simulation.engines import offline_table as offline_mod
from tricys.simulation.engines import standard as standard_mod
from tricys.simulation.engines.offline_table import OfflineTableEngine
from tricys.simulation.engines.standard import StandardSimulationEngine
from tricys.workflow.base import JobExecutionError
from tricys.workflow.context import PipelineContext

pytestmark = pytest.mark.build_test


def _context(tmp_path, n_jobs: int, *, concurrent: bool = False) -> PipelineContext:
    config = {
        "paths": {"package_path": str(tmp_path / "model.mo")},
        "simulation": {
            "stop_time": 1.0,
            "step_size": 0.1,
            "concurrent": concurrent,
        },
    }
    return PipelineContext(
        config=config,
        base_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        results_dir=str(tmp_path / "results"),
        package_path=str(tmp_path / "model.mo"),
        model_name="Example",
        jobs=[{} for _ in range(n_jobs)],
        compiled_artifacts={
            "exe": str(tmp_path / "model"),
            "xml": str(tmp_path / "model_init.xml"),
            "om_bin": "/usr/bin",
        },
    )


class _FlakyModelicaSystem:
    """Fails every ``simulate`` call starting from ``fail_from`` (1-based)."""

    calls = 0
    fail_from: int | None = None

    def __init__(self, *args, **kwargs):
        pass

    def setSimulationOptions(self, options):
        pass

    def setParameters(self, params):
        pass

    def simulate(self, resultfile=None, simargs=None):
        type(self).calls += 1
        if type(self).fail_from is not None and type(self).calls >= type(self).fail_from:
            raise RuntimeError("solver diverged")
        pd.DataFrame({"time": [0.0, 1.0], "x": [1.0, 2.0]}).to_csv(
            resultfile, index=False
        )


def _patch_sequential(monkeypatch):
    class _FakeSession:
        def sendExpression(self, expression):
            return "ok"

    monkeypatch.setattr(standard_mod, "get_om_session", lambda: _FakeSession())
    monkeypatch.setattr(standard_mod, "load_modelica_package", lambda *a, **k: True)
    monkeypatch.setattr(standard_mod, "ModelicaSystem", _FlakyModelicaSystem)


def _patch_concurrent(monkeypatch, failing_job: int):
    def _fake_worker(args):
        job_params, job_id, kwargs = args
        if job_id == failing_job:
            return job_id, job_params, "", "solver diverged"
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

    monkeypatch.setattr(standard_mod, "mp_fast_job_worker", _fake_worker)
    monkeypatch.setattr(standard_mod.multiprocessing, "Pool", _FakePool)


@pytest.fixture(autouse=True)
def _reset_flaky():
    _FlakyModelicaSystem.calls = 0
    _FlakyModelicaSystem.fail_from = None
    yield
    _FlakyModelicaSystem.calls = 0
    _FlakyModelicaSystem.fail_from = None


# --------------------------------------------------------------------------- standard


def test_sequential_all_jobs_succeed(monkeypatch, tmp_path):
    _patch_sequential(monkeypatch)
    engine = StandardSimulationEngine(parallel=False)
    engine.execute(_context(tmp_path, 2))  # must not raise


def test_sequential_any_job_failure_raises(monkeypatch, tmp_path):
    _patch_sequential(monkeypatch)
    _FlakyModelicaSystem.fail_from = 2  # job 1 ok, jobs 2 and 3 fail

    with pytest.raises(JobExecutionError) as excinfo:
        StandardSimulationEngine(parallel=False).execute(_context(tmp_path, 3))

    error = excinfo.value
    assert error.failed == 2
    assert error.total == 3
    assert {job_id for job_id, _ in error.failures} == {2, 3}


def test_concurrent_all_jobs_succeed(monkeypatch, tmp_path):
    _patch_concurrent(monkeypatch, failing_job=-1)
    StandardSimulationEngine(parallel=True).execute(_context(tmp_path, 2, concurrent=True))


def test_concurrent_any_job_failure_raises(monkeypatch, tmp_path):
    _patch_concurrent(monkeypatch, failing_job=2)

    with pytest.raises(JobExecutionError) as excinfo:
        StandardSimulationEngine(parallel=True).execute(
            _context(tmp_path, 3, concurrent=True)
        )

    error = excinfo.value
    assert error.failed == 1
    assert error.total == 3
    assert error.failures[0][0] == 2


# ----------------------------------------------------------------------- co-simulation


def test_cosim_any_job_failure_raises(monkeypatch, tmp_path):
    def _fake_run(config, job_params, job_id=0):
        return "" if job_id == 2 else str(tmp_path / f"job_{job_id}.csv")

    monkeypatch.setattr(offline_mod, "run_co_simulation_job", _fake_run)

    with pytest.raises(JobExecutionError) as excinfo:
        OfflineTableEngine().execute(_context(tmp_path, 3))

    error = excinfo.value
    assert error.failed == 1
    assert error.failures[0][0] == 2


# ------------------------------------------------------------------------- exit code


def test_cli_exit_code_is_nonzero_on_job_failure(monkeypatch):
    """The contract that users and CI actually see: exit code 1, not a green run."""
    monkeypatch.setattr(
        simulation_mod, "basic_prepare_config", lambda *a, **k: ({}, {})
    )
    monkeypatch.setattr(simulation_mod, "setup_logging", lambda *a, **k: None)

    def _boom(*args, **kwargs):
        raise JobExecutionError("standard", [(1, "solver diverged")], 1)

    monkeypatch.setattr(simulation_mod, "run_simulation", _boom)

    with pytest.raises(SystemExit) as excinfo:
        simulation_mod.main("config.json")

    assert excinfo.value.code == 1
