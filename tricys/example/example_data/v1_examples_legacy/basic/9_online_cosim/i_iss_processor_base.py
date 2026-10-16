"""
Auto-generated strongly typed processor base class for I_ISS, enhanced with Aspen logic.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional

from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)

logger = logging.getLogger(__name__)


@dataclass
class I_ISSInputData:
    from_TEP_FCU: List[float] = field(default_factory=list)


@dataclass
class I_ISSOutputData:
    to_SDS: Optional[List[float]] = None
    to_WDS: Optional[List[float]] = None


TRITIUM_MOLAR_MASS_G_PER_MOL = 3.016
DEUTERIUM_MOLAR_MASS_G_PER_MOL = 2.014
HYDROGEN_MOLAR_MASS_G_PER_MOL = 1.008


def build_i_iss_aspen_ratios(in_mass_flows: list[float]) -> tuple[list[float], float]:
    """Convert T/D/H mass flows into Aspen six-species ratios and total molar flow."""
    if len(in_mass_flows) < 3:
        raise ValueError(
            "I_ISS Aspen step input must contain at least T/D/H mass flows"
        )

    tritium_flow_gph = max(float(in_mass_flows[0]), 0.0)
    deuterium_flow_gph = max(float(in_mass_flows[1]), 0.0)
    hydrogen_flow_gph = max(float(in_mass_flows[2]), 0.0)

    tritium_molph = tritium_flow_gph / TRITIUM_MOLAR_MASS_G_PER_MOL
    deuterium_molph = deuterium_flow_gph / DEUTERIUM_MOLAR_MASS_G_PER_MOL
    hydrogen_molph = hydrogen_flow_gph / HYDROGEN_MOLAR_MASS_G_PER_MOL
    total_molph = tritium_molph + deuterium_molph + hydrogen_molph

    if total_molph <= 0.0:
        return [0.0] * 7, 0.0

    et = tritium_molph / total_molph
    ed = deuterium_molph / total_molph
    eh = hydrogen_molph / total_molph

    ratios = [
        eh**2,
        2 * eh * ed,
        ed**2,
        2 * eh * et,
        2 * ed * et,
        et**2,
        total_molph,
    ]
    return ratios, total_molph


def map_i_iss_stream_results(
    stream_results: Mapping[str, list[float]],
) -> tuple[list[float], list[float]]:
    """Map Aspen stream outputs into I_ISS online track outputs."""
    sdst2 = list(stream_results.get("SDST2", [0.0, 0.0, 0.0]))
    sdsd2 = list(stream_results.get("SDSD2", [0.0, 0.0, 0.0]))
    wds = list(stream_results.get("WDS", [0.0, 0.0, 0.0]))

    to_sds = [
        sdst2[2] + sdsd2[2],
        sdst2[1] + sdsd2[1],
        sdst2[0] + sdsd2[0],
        0.0,
        0.0,
    ]
    to_wds = [wds[2], wds[1], wds[0], 0.0, 0.0]

    return to_sds, to_wds


class I_ISSProcessorBase(AbstractTrackProcessor):
    def __init__(
        self,
        bkp_path: str = "mock_path",
        aspen_factory: Callable[[str], Any] | None = None,
        base_inventory_mol: float = 20.0,
        retime_h: float = 1.0,
    ):
        self.bkp_path = bkp_path
        self.aspen_factory = aspen_factory
        self.base_inventory_mol = float(base_inventory_mol)
        self.retime_h = float(retime_h)
        self._aspen = None
        self.inventory_mol = [0.0, 0.0, 0.0]  # T, D, H
        self.delay_queue: list[
            tuple[float, list[float], list[float], list[float], float]
        ] = []
        self.in_transit_mass_g = 0.0
        self.in_transit_T_mass_g = 0.0
        self._current_decay_rate = [0.0] * 5

    def initialize(self, context: TrackProcessorContext) -> None:
        self.inventory_mol = [0.0, 0.0, 0.0]
        self.delay_queue = []
        self.in_transit_T_mass_g = 0.0

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        if "MOCK" in self.bkp_path:
            logger.info(
                "MOCK FALLBACK TRIGGERED: Falling back to Shadow FMU execution as requested by 'MOCK' configuration."
            )

        inputs = I_ISSInputData(
            from_TEP_FCU=request_vector.boundary_inputs.get("from_TEP_FCU", []),
        )
        outputs = I_ISSOutputData()
        extra_state = request_vector.extra_state or {}

        self.on_step(
            request_vector.current_time_h,
            request_vector.dt_slow_h,
            inputs,
            outputs,
            extra_state,
        )

        out_dict = {}
        if outputs.to_SDS is not None:
            out_dict["to_SDS"] = outputs.to_SDS
        if outputs.to_WDS is not None:
            out_dict["to_WDS"] = outputs.to_WDS

        return TrackResult(outputs=out_dict)

    def on_step(
        self,
        current_time_h: float,
        dt_slow_h: float,
        inputs: I_ISSInputData,
        outputs: I_ISSOutputData,
        extra_state: dict,
    ) -> None:
        if dt_slow_h <= 0.0:
            raise ValueError(
                "dt_slow_h must be positive for Aspen single-step execution"
            )

        step_input = inputs.from_TEP_FCU
        if step_input and len(step_input) >= 3:
            tritium_molph = (
                max(float(step_input[0]), 0.0) / TRITIUM_MOLAR_MASS_G_PER_MOL
            )
            deuterium_molph = (
                max(float(step_input[1]), 0.0) / DEUTERIUM_MOLAR_MASS_G_PER_MOL
            )
            hydrogen_molph = (
                max(float(step_input[2]), 0.0) / HYDROGEN_MOLAR_MASS_G_PER_MOL
            )
        else:
            tritium_molph = deuterium_molph = hydrogen_molph = 0.0

        self.inventory_mol[0] += tritium_molph * dt_slow_h
        self.inventory_mol[1] += deuterium_molph * dt_slow_h
        self.inventory_mol[2] += hydrogen_molph * dt_slow_h

        # Apply radioactive decay to inventory
        decay_constant_T = 6.4e-6  # per second
        decay_rate_g = [0.0] * 5

        t_mass_g = self.inventory_mol[0] * TRITIUM_MOLAR_MASS_G_PER_MOL
        decay_rate_g[0] = t_mass_g * decay_constant_T
        decay_mass_g = decay_rate_g[0] * (dt_slow_h * 3600.0)
        self.inventory_mol[0] = max(
            0.0, self.inventory_mol[0] - (decay_mass_g / TRITIUM_MOLAR_MASS_G_PER_MOL)
        )

        # Convert from g/s to g/h for auditor
        decay_rate_g_h = [rate * 3600.0 for rate in decay_rate_g]
        self._current_decay_rate = decay_rate_g_h

        i_total = sum(self.inventory_mol)

        if i_total <= self.base_inventory_mol:
            self.delay_queue.append(
                (current_time_h + self.retime_h, [0.0] * 5, [0.0] * 5, [0.0] * 5, 0.0)
            )
            self._pop_from_queue(current_time_h, outputs)
            return

        # Overflow logic: Release all inventory above base_inventory_mol
        excess_moles = i_total - self.base_inventory_mol
        effective_total_molph = excess_moles / dt_slow_h

        if effective_total_molph <= 0.0:
            self.delay_queue.append(
                (current_time_h + self.retime_h, [0.0] * 5, [0.0] * 5, [0.0] * 5, 0.0)
            )
            self._pop_from_queue(current_time_h, outputs)
            return

        # Build Aspen ratios based on the tank's composition
        et = self.inventory_mol[0] / i_total
        ed = self.inventory_mol[1] / i_total
        eh = self.inventory_mol[2] / i_total

        ratios = [
            eh**2,
            2 * eh * ed,
            ed**2,
            2 * eh * et,
            2 * ed * et,
            et**2,
            effective_total_molph,
        ]

        aspen = self._get_aspen()
        aspen.set_composition(ratios)
        aspen.run_step()
        stream_results = aspen.get_stream_results()

        primary, secondary = map_i_iss_stream_results(stream_results)

        # Deduct exactly what we sent to Aspen from inventory.
        self.inventory_mol[0] -= excess_moles * et
        self.inventory_mol[1] -= excess_moles * ed
        self.inventory_mol[2] -= excess_moles * eh

        t_mass_sent = (excess_moles * et) * TRITIUM_MOLAR_MASS_G_PER_MOL
        self.in_transit_T_mass_g += t_mass_sent

        # Calculate exactly what Aspen lost and send it to VDS
        flow_in_T = (excess_moles * et / dt_slow_h) * TRITIUM_MOLAR_MASS_G_PER_MOL
        flow_in_D = (excess_moles * ed / dt_slow_h) * DEUTERIUM_MOLAR_MASS_G_PER_MOL
        flow_in_H = (excess_moles * eh / dt_slow_h) * HYDROGEN_MOLAR_MASS_G_PER_MOL

        flow_out_T = primary[0] + secondary[0]
        flow_out_D = primary[1] + secondary[1]
        flow_out_H = primary[2] + secondary[2]

        to_vds = [
            max(0.0, flow_in_T - flow_out_T),
            max(0.0, flow_in_D - flow_out_D),
            max(0.0, flow_in_H - flow_out_H),
            0.0,
            0.0,
        ]

        self.delay_queue.append(
            (current_time_h + self.retime_h, primary, secondary, to_vds, t_mass_sent)
        )
        self._pop_from_queue(current_time_h, outputs)

    def _pop_from_queue(self, current_time_h: float, outputs: I_ISSOutputData) -> None:
        ready_primary, ready_secondary, ready_vds = [0.0] * 5, [0.0] * 5, [0.0] * 5
        while self.delay_queue and current_time_h >= self.delay_queue[0][0]:
            _, ready_primary, ready_secondary, ready_vds, t_mass = self.delay_queue.pop(
                0
            )
            self.in_transit_T_mass_g = max(0.0, self.in_transit_T_mass_g - t_mass)

        outputs.to_SDS = ready_primary
        outputs.to_WDS = ready_secondary

    def finalize(self) -> None:
        if self._aspen is not None and hasattr(self._aspen, "close"):
            self._aspen.close()
        self._aspen = None
        self.on_finalize()

    def on_finalize(self) -> None:
        pass

    def get_mass_inventory(self) -> float:
        mass = self.inventory_mol[0] * TRITIUM_MOLAR_MASS_G_PER_MOL
        return mass + self.in_transit_T_mass_g

    def get_decay_rate(self) -> float:
        return float(self._current_decay_rate[0])

    def get_release_rate(self) -> float:
        return 0.0

    def _get_aspen(self):
        if self._aspen is None:
            factory = self.aspen_factory or self._load_default_aspen_factory()
            self._aspen = factory(self.bkp_path)
        return self._aspen

    @staticmethod
    def _load_default_aspen_factory() -> Callable[[str], Any]:
        from tricys.handlers.i_iss_handler import AspenEnhanced

        return AspenEnhanced
