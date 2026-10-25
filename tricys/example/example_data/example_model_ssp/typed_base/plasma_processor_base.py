"""
Auto-generated strongly typed processor base class for Plasma.
Do not edit manually. Re-run generate_processors.py to update.
"""

from dataclasses import dataclass
from typing import List, Optional

from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)


@dataclass
class PlasmaInputData:
    pulseInput: float = 0.0


@dataclass
class PlasmaOutputData:
    from_Fueling_System: Optional[List[float]] = None
    to_FW: Optional[List[float]] = None
    to_Div: Optional[List[float]] = None
    to_Pump: Optional[List[float]] = None


class PlasmaProcessorBase(AbstractTrackProcessor):
    def initialize(self, context: TrackProcessorContext) -> None:
        self.on_initialize(context)

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        inputs = PlasmaInputData(
            pulseInput=request_vector.boundary_inputs.get("pulseInput", 0.0),
        )
        outputs = PlasmaOutputData()
        extra_state = request_vector.extra_state or {}

        self.on_step(
            request_vector.current_time_h,
            request_vector.dt_slow_h,
            inputs,
            outputs,
            extra_state,
        )

        out_dict = {}
        if outputs.from_Fueling_System is not None:
            out_dict["from_Fueling_System"] = outputs.from_Fueling_System
        if outputs.to_FW is not None:
            out_dict["to_FW"] = outputs.to_FW
        if outputs.to_Div is not None:
            out_dict["to_Div"] = outputs.to_Div
        if outputs.to_Pump is not None:
            out_dict["to_Pump"] = outputs.to_Pump

        return TrackResult(outputs=out_dict)

    def finalize(self) -> None:
        self.on_finalize()

    def on_initialize(self, context: TrackProcessorContext) -> None:
        pass

    def on_step(
        self,
        current_time_h: float,
        dt_slow_h: float,
        inputs: PlasmaInputData,
        outputs: PlasmaOutputData,
        extra_state: dict,
    ) -> None:
        raise NotImplementedError("Subclasses must implement on_step")

    def on_finalize(self) -> None:
        pass

    def get_mass_inventory(self) -> float:
        """
        Report the total mass (in grams) currently buffered, delayed, or
        held inside this processor and its external surrogate.
        Returns 0.0 by default.
        """
        return 0.0

    def get_decay_rate(self) -> float:
        """
        Report the radioactive decay rate for Tritium (in g/s).
        Returns 0.0 by default.
        """
        return 0.0

    def get_release_rate(self) -> float:
        """
        Report the environmental release rate for Tritium (in g/s).
        Returns 0.0 by default.
        """
        return 0.0
