"""
Auto-generated strongly typed processor base class for Coolant_Pipe.
Do not edit manually. Re-run generate_processors.py to update.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)


@dataclass
class Coolant_PipeInputData:
    from_FW: List[float] = field(default_factory=list)
    from_DIV: List[float] = field(default_factory=list)
    from_BZ: List[float] = field(default_factory=list)


@dataclass
class Coolant_PipeOutputData:
    to_CPS: Optional[List[float]] = None
    to_FW: Optional[List[float]] = None
    to_DIV: Optional[List[float]] = None
    to_WDS: Optional[List[float]] = None


class Coolant_PipeProcessorBase(AbstractTrackProcessor):
    def initialize(self, context: TrackProcessorContext) -> None:
        self.on_initialize(context)

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        inputs = Coolant_PipeInputData(
            from_FW=request_vector.boundary_inputs.get("from_FW", []),
            from_DIV=request_vector.boundary_inputs.get("from_DIV", []),
            from_BZ=request_vector.boundary_inputs.get("from_BZ", []),
        )
        outputs = Coolant_PipeOutputData()
        extra_state = request_vector.extra_state or {}

        self.on_step(
            request_vector.current_time_h,
            request_vector.dt_slow_h,
            inputs,
            outputs,
            extra_state,
        )

        out_dict = {}
        if outputs.to_CPS is not None:
            out_dict["to_CPS"] = outputs.to_CPS
        if outputs.to_FW is not None:
            out_dict["to_FW"] = outputs.to_FW
        if outputs.to_DIV is not None:
            out_dict["to_DIV"] = outputs.to_DIV
        if outputs.to_WDS is not None:
            out_dict["to_WDS"] = outputs.to_WDS

        return TrackResult(outputs=out_dict)

    def finalize(self) -> None:
        self.on_finalize()

    def on_initialize(self, context: TrackProcessorContext) -> None:
        pass

    def on_step(
        self,
        current_time_h: float,
        dt_slow_h: float,
        inputs: Coolant_PipeInputData,
        outputs: Coolant_PipeOutputData,
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
