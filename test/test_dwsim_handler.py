# -*- coding: utf-8 -*-
"""Unit tests for DWSIM ISS co-simulation handler."""

import importlib.util
import os
from pathlib import Path
import tempfile
import pandas as pd
import pytest

_HANDLER_PATH = (
    Path(__file__).resolve().parent.parent
    / "tricys"
    / "example"
    / "example_data"
    / "v2_examples"
    / "08_cosim_offline_table"
    / "dwsim_iss_handler.py"
)
_spec = importlib.util.spec_from_file_location("dwsim_iss_handler", _HANDLER_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

DwsimEnhanced = _module.DwsimEnhanced
run_dwsim_simulation = _module.run_dwsim_simulation


def test_dwsim_enhanced_surrogate_fallback():
    """Verify DwsimEnhanced surrogate model computes reasonable isotopic separation."""
    handler = DwsimEnhanced(flowsheet_path="non_existent_flowsheet.dwxmz")
    assert handler.is_fallback is True

    # Feed: 100 mol/h with equal H, D, T
    # EH = 1/3, ED = 1/3, ET = 1/3
    eh, ed, et = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    ratios = [
        eh**2,
        2 * eh * ed,
        ed**2,
        2 * eh * et,
        2 * ed * et,
        et**2,
        100.0,
    ]
    handler.set_composition(ratios)
    handler.run_step()
    res = handler.get_stream_results()

    assert "WDS" in res
    assert "SDST2" in res
    assert "SDSD2" in res

    # WDS should be predominantly Protium (H)
    assert res["WDS"][0] > res["WDS"][2]
    # SDST2 should be predominantly Tritium (T)
    assert res["SDST2"][2] > res["SDST2"][0]
    # SDSD2 should be predominantly Deuterium (D)
    assert res["SDSD2"][1] > res["SDSD2"][0]

    handler.close()


def test_run_dwsim_simulation_offline_table():
    """Verify run_dwsim_simulation runs 2-stage offline-table workflow properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "primary_inputs.csv")
        output_csv = os.path.join(tmpdir, "dwsim_outputs.csv")

        # Create mock primary input data from OpenModelica
        # time in hours: 0.0, 0.05, 0.1, ... 2.0
        times = [i * 0.05 for i in range(41)]
        df_in = pd.DataFrame(
            {
                "time": times,
                "tep_fcu.outflow[1]": [10.0] * len(times),  # T (g/h)
                "tep_fcu.outflow[2]": [20.0] * len(times),  # D (g/h)
                "tep_fcu.outflow[3]": [5.0] * len(times),  # H (g/h)
            }
        )
        df_in.to_csv(input_csv, index=False)

        res_mapping = run_dwsim_simulation(
            temp_input_csv=input_csv,
            temp_output_csv=output_csv,
            flowsheet_path="mock.dwxmz",
            base=5.0,
            retime=6,  # 6 min delay
            time_step=3,  # 3 min step
            min_stable_steps=50,
        )

        assert os.path.exists(output_csv)
        assert "to_SDS" in res_mapping
        assert "to_WDS" in res_mapping
        assert res_mapping["to_SDS"] == "{1,2,3,4,1,1}"
        assert res_mapping["to_WDS"] == "{1,5,6,7,1,1}"

        # Modelica CombiTimeTable with tableOnFile=true requires pure numerical data (header=False)
        out_df = pd.read_csv(output_csv, header=None)
        assert out_df.shape[1] == 7  # Time + 3 to_SDS ports + 3 to_WDS ports
        assert len(out_df) > 0
        # Check first column is time
        assert out_df.iloc[0, 0] == 0.0
