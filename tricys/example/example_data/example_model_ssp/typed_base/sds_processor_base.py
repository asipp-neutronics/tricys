"""
Auto-generated strongly typed processor base class for SDS.
Do not edit manually. Re-run generate_processors.py to update.
"""

from dataclasses import dataclass, field
from typing import List

from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)


@dataclass
class SDSInputData:
    from_I_ISS: List[float] = field(default_factory=list)
    from_O_ISS: List[float] = field(default_factory=list)
    from_TEP_FEP: List[float] = field(default_factory=list)
    to_FS: List[float] = field(default_factory=list)


@dataclass
class SDSOutputData:
    pass


class SDSProcessorBase(AbstractTrackProcessor):
    def initialize(self, context: TrackProcessorContext) -> None:
        self.on_initialize(context)

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        inputs = SDSInputData(
            from_I_ISS=request_vector.boundary_inputs.get("from_I_ISS", []),
            from_O_ISS=request_vector.boundary_inputs.get("from_O_ISS", []),
            from_TEP_FEP=request_vector.boundary_inputs.get("from_TEP_FEP", []),
            to_FS=request_vector.boundary_inputs.get("to_FS", []),
        )
        outputs = SDSOutputData()
        extra_state = request_vector.extra_state or {}

        self.on_step(
            request_vector.current_time_h,
            request_vector.dt_slow_h,
            inputs,
            outputs,
            extra_state,
        )

        out_dict = {}

        return TrackResult(outputs=out_dict)

    def finalize(self) -> None:
        self.on_finalize()

    def on_initialize(self, context: TrackProcessorContext) -> None:
        pass

    def on_step(
        self,
        current_time_h: float,
        dt_slow_h: float,
        inputs: SDSInputData,
        outputs: SDSOutputData,
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
