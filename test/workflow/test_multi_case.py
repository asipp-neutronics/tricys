from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tricys.analysis.cases import AnalysisCasesManager
from tricys.simulation.simulation import run_simulation


def test_analysis_cases_manager_is_analysis_cases():
    assert AnalysisCasesManager.is_analysis_cases(
        {"sensitivity_analysis": {"analysis_cases": [{"name": "c1"}]}}
    )
    assert not AnalysisCasesManager.is_analysis_cases(
        {"sensitivity_analysis": {"analysis_cases": []}}
    )
    assert not AnalysisCasesManager.is_analysis_cases({})


def test_analysis_cases_manager_deprecation_and_forwarding():
    config = {
        "sensitivity_analysis": {
            "analysis_cases": [{"name": "CaseA"}],
        }
    }

    with patch("tricys.simulation.simulation.run_simulation") as mock_run_sim:
        with pytest.deprecated_call(match="AnalysisCasesManager is deprecated"):
            handled = AnalysisCasesManager.handle_analysis_cases(config)
            assert handled is True
            mock_run_sim.assert_called_once_with(config)


def test_run_simulation_dispatches_single_pipeline():
    config = {
        "paths": {"temp_dir": "temp", "results_dir": "results"},
        "simulation": {"model_name": "TestModel"},
        "sensitivity_analysis": {
            "analysis_cases": [{"name": "SingleCase"}],
        },
    }

    with (
        patch("tricys.simulation.simulation._run_single_pipeline") as mock_single,
        patch("tricys.simulation.simulation._run_multi_case_pipeline") as mock_multi,
    ):
        mock_single.return_value = MagicMock()
        _ = run_simulation(config)
        mock_single.assert_called_once()
        mock_multi.assert_not_called()


def test_run_simulation_dispatches_multi_pipeline():
    config = {
        "paths": {"temp_dir": "temp", "results_dir": "results"},
        "simulation": {"model_name": "TestModel"},
        "sensitivity_analysis": {
            "analysis_cases": [{"name": "Case1"}, {"name": "Case2"}],
        },
    }

    with (
        patch("tricys.simulation.simulation._run_single_pipeline") as mock_single,
        patch("tricys.simulation.simulation._run_multi_case_pipeline") as mock_multi,
    ):
        mock_multi.return_value = [MagicMock(), MagicMock()]
        _ = run_simulation(config)
        mock_multi.assert_called_once()
        mock_single.assert_not_called()
