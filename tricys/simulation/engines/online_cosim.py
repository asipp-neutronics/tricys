from __future__ import annotations

import logging

from tricys.workflow.base import BaseSimulationEngine
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_simulation_engine

logger = logging.getLogger(__name__)


@register_simulation_engine("online_oms")
class OnlineOmsCoSimEngine(BaseSimulationEngine):
    """Core simulation engine for online OMSimulator / FMI / SSP co-simulation bus."""

    name = "online_oms"

    def execute(self, context: PipelineContext) -> None:
        from tricys.online_cosim.runtime import run_online_cosimulation

        logger.info("OnlineOmsCoSimEngine: Starting OMSimulator online co-simulation")
        res_dict = run_online_cosimulation(context.config, export_csv=False)

        result_file = res_dict.get("result_file")
        hdf_path = res_dict.get("hdf_path")

        if hdf_path:
            context.results_store_path = hdf_path

        context.raw_results = [
            {
                "job_id": 1,
                "params": {},
                "result_path": result_file,
                "hdf_path": hdf_path,
                "online_res": res_dict,
            }
        ]
        context.set_metadata("online_cosim_result", res_dict)
        logger.info(
            f"OnlineOmsCoSimEngine: Finished successfully. Output: {result_file}"
        )
