from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(slots=True)
class UnifiedStateVector:
    """Normalized request payload for one online co-simulation step."""

    component_name: str
    step_id: int
    seq_id: int
    current_time_h: float
    dt_slow_h: float
    boundary_inputs: Dict[str, Any]
    extra_state: Dict[str, Any] | None = None

    # Diagnostic metric from OnlineGlobalAuditor. Reports the mass balance
    # residual (grams) computed at the *previous* audit step.  Processors
    # may log or record this value but should NOT use it to alter their
    # physical outputs.  Active compensation is handled directly by the
    # auditor's PI controller writing to the SDS compensation_flow port.
    global_mass_error: float = 0.0

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "component_name": self.component_name,
            "step_id": self.step_id,
            "seq_id": self.seq_id,
            "current_time_h": self.current_time_h,
            "dt_slow_h": self.dt_slow_h,
            "boundary_inputs": dict(self.boundary_inputs),
            "global_mass_error": self.global_mass_error,
        }
        if self.extra_state is not None:
            payload["extra_state"] = dict(self.extra_state)
        return payload


@dataclass(slots=True)
class TrackResult:
    """Normalized processor response for one online co-simulation step."""

    outputs: Dict[str, Any]
    fallback_to_fmu: bool = False

    def __post_init__(self) -> None:
        normalized_outputs = {
            str(name): self._normalize_output_value(value)
            for name, value in dict(self.outputs).items()
        }
        object.__setattr__(self, "outputs", normalized_outputs)

    @staticmethod
    def _normalize_output_value(value: Any) -> Any:
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        return value


@dataclass(slots=True)
class TrackProcessorContext:
    """Execution context provided to one track processor instance."""

    processor_name: str
    processor_index: int
    run_id: str | None = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SignalBinding:
    """Describe one scalar or vector signal mapped into OMS runtime."""

    component_name: str
    signal_name: str
    width: int = 1

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("signal binding width must be at least 1")

    @property
    def is_vector(self) -> bool:
        return self.width > 1


@dataclass(frozen=True, slots=True)
class OmsStepDescriptor:
    """Describe one processor step coupled with one OMS runtime advance."""

    component_name: str
    step_id: int
    seq_id: int
    current_time_h: float
    dt_slow_h: float
    target_time_h: float
    input_bindings: Dict[str, SignalBinding]
    output_bindings: Dict[str, SignalBinding] = field(default_factory=dict)
    extra_state_bindings: Dict[str, SignalBinding] = field(default_factory=dict)
    extra_state: Dict[str, Any] | None = None
    processor_index: int = 0

    def __post_init__(self) -> None:
        if self.processor_index < 0:
            raise ValueError("processor_index must be non-negative")
        if self.dt_slow_h <= 0.0:
            raise ValueError("dt_slow_h must be positive")
        if self.target_time_h < self.current_time_h:
            raise ValueError(
                "target_time_h must be greater than or equal to current_time_h"
            )
