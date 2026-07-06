from abc import ABC, abstractmethod
from typing import Any, Mapping

from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)


def normalize_output_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_output_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("processor outputs must be a mapping")
    return {
        str(name): normalize_output_list(output_value)
        for name, output_value in value.items()
    }


class AbstractTrackProcessor(ABC):
    """Minimal lifecycle contract for one online co-simulation processor."""

    @abstractmethod
    def initialize(self, context: TrackProcessorContext) -> None:
        """Prepare the processor for subsequent `process` calls."""

    @abstractmethod
    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        """Process one communication step and return normalized outputs."""

    @abstractmethod
    def finalize(self) -> None:
        """Release processor resources after the run finishes."""

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
