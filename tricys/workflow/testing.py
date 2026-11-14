from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from tricys.workflow.context import PipelineContext, SimulationJobResult


def create_mock_context(
    num_jobs: int = 1,
    stop_time: float = 1000.0,
    step_size: float = 10.0,
    base_storage_g: float = 6000.0,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineContext:
    """Creates an in-memory mock PipelineContext populated with realistic fusion timeseries data.

    Enables Agent and developers to execute and verify custom Preprocess/Postprocess plugins
    within a millisecond-level sandbox without invoking actual OpenModelica compilation.
    """
    temp_dir = tempfile.mkdtemp(prefix="tricys_mock_")
    results_dir = os.path.join(temp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    base_config: Dict[str, Any] = {
        "version": "2.0",
        "model": {
            "fileName": "mock_model.mo",
            "modelName": "MockCycle.Cycle",
        },
        "scenario": {
            "stopTime": stop_time,
            "stepSize": step_size,
        },
        "task": {
            "type": "simulate" if num_jobs == 1 else "sweep",
            "parameters": {"sds.V": 1.5},
        },
        "workspace": {
            "results_dir": results_dir,
            "temp_dir": temp_dir,
        },
    }
    if config_overrides:
        base_config.update(config_overrides)

    context = PipelineContext.from_config(base_config, base_dir=temp_dir)
    context.results_dir = results_dir
    context.temp_dir = temp_dir

    # Generate synthetic realistic fusion fuel cycle timeseries for each job
    t_vals = np.arange(0.0, stop_time + step_size, step_size)
    mock_results: List[SimulationJobResult] = []
    mock_raw_results: List[Dict[str, Any]] = []

    for j_id in range(1, num_jobs + 1):
        # Synthetic inventory: initial dip down to valley, then turnaround
        decay_factor = np.exp(-t_vals / (12.3 * 365 * 24 * 3600))
        startup_dip = -400.0 * (1.0 - np.exp(-t_vals / 100.0))
        tbr_growth = 0.05 * t_vals
        sds_inv = (base_storage_g + startup_dip + tbr_growth) * decay_factor

        # Build realistic DataFrame with typical Modelica polymorphic columns
        df = pd.DataFrame(
            {
                "time": t_vals,
                "Cycle_3LC.sds.inventory": sds_inv,
                "sds.inventory": sds_inv,
                "blanket.inventory": 150.0 + 10.0 * np.sin(t_vals / 50.0),
                "coolant_pipe.inventory": 80.0 + 5.0 * np.cos(t_vals / 50.0),
                "plasma.fb": np.full_like(t_vals, 0.03),
                "plasma.W_in": np.full_like(t_vals, 15.0),
                "tep_fep.inventory": 45.0 + 2.0 * np.sin(t_vals / 30.0),
            }
        )

        csv_path = os.path.join(temp_dir, f"mock_job_{j_id}_res.csv")
        df.to_csv(csv_path, index=False)

        job_params = {"job_id": j_id, "sds.V": 1.5 + (j_id - 1) * 0.2}

        # Typed contract
        job_result = SimulationJobResult(
            job_id=j_id,
            parameters=job_params,
            dataframe=df,
            result_path=csv_path,
            time_column="time",
            time_unit="second",
            metrics={"min_inventory": float(sds_inv.min())},
            extra={"mock": True},
        )
        mock_results.append(job_result)

        # Legacy raw results
        mock_raw_results.append(
            {
                "job_id": j_id,
                "params": job_params,
                "result_path": csv_path,
                "dataframe": df,
            }
        )

    context.results = mock_results
    context.raw_results = mock_raw_results

    return context
