"""
Auto-generated strongly typed processor base class for CPS.
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
class CPSInputData:
    from_CL: List[float] = field(default_factory=list)


@dataclass
class CPSOutputData:
    to_ISS_O: Optional[List[float]] = None
    to_FW: Optional[List[float]] = None
    to_DIV: Optional[List[float]] = None


class CPSProcessorBase(AbstractTrackProcessor):
    def initialize(self, context: TrackProcessorContext) -> None:
        self.on_initialize(context)

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        inputs = CPSInputData(
            from_CL=request_vector.boundary_inputs.get("from_CL", []),
        )
        outputs = CPSOutputData()
        extra_state = request_vector.extra_state or {}

        self.on_step(
            request_vector.current_time_h,
            request_vector.dt_slow_h,
            inputs,
            outputs,
            extra_state,
        )

        out_dict = {}
        if outputs.to_ISS_O is not None:
            out_dict["to_ISS_O"] = outputs.to_ISS_O
        if outputs.to_FW is not None:
            out_dict["to_FW"] = outputs.to_FW
        if outputs.to_DIV is not None:
            out_dict["to_DIV"] = outputs.to_DIV

        return TrackResult(outputs=out_dict)

    def finalize(self) -> None:
        self.on_finalize()

    def on_initialize(self, context: TrackProcessorContext) -> None:
        pass

    def on_step(
        self,
        current_time_h: float,
        dt_slow_h: float,
        inputs: CPSInputData,
        outputs: CPSOutputData,
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
