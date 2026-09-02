from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from tricys.online_cosim.schema import SignalBinding


class OmsBindingsError(RuntimeError):
    """Raised when OMSimulator bindings are unavailable."""


@dataclass(frozen=True, slots=True)
class OmsBindings:
    """Minimal set of OMSimulator factories required by the runtime wrapper."""

    ssp_factory: Callable[[], Any]
    cref_factory: Callable[..., Any]


def load_oms_bindings() -> OmsBindings:
    """Load OMSimulator factories lazily.

    The import is delayed so Phase 2 unit tests can exercise the runtime with
    fake bindings even when OMSimulator is not installed in the environment.
    """

    try:
        from OMSimulator import SSP, CRef
    except ImportError as exc:
        raise OmsBindingsError(
            "OMSimulator Python bindings are required for online OMS runtime"
        ) from exc

    return OmsBindings(ssp_factory=SSP, cref_factory=CRef)


class OmsSystemRuntime:
    """Thin, mockable wrapper around an OMSimulator system instance."""

    def __init__(
        self,
        bindings: OmsBindings | None = None,
        model: Any | None = None,
    ) -> None:
        self._bindings = bindings or load_oms_bindings()
        self._model = model if model is not None else self._bindings.ssp_factory()
        self._instance: Any | None = None

    @property
    def model(self) -> Any:
        return self._model

    @property
    def instance(self) -> Any | None:
        return self._instance

    def cref(
        self, system_name: str, component_name: str, variable_name: str | None = None
    ) -> Any:
        if variable_name is None:
            return self._bindings.cref_factory(system_name, component_name)
        return self._bindings.cref_factory(system_name, component_name, variable_name)

    def add_resource(self, resource_path: str, new_name: str) -> None:
        self._model.addResource(resource_path, new_name=new_name)

    def add_component(
        self, system_name: str, component_name: str, resource_name: str
    ) -> None:
        self._model.addComponent(self.cref(system_name, component_name), resource_name)

    def add_connection(self, source_cref: Any, target_cref: Any) -> None:
        self._model.addConnection(source_cref, target_cref)

    def instantiate(self) -> Any:
        self._instance = self._model.instantiate()
        return self._instance

    def _require_instance(self) -> Any:
        if self._instance is None:
            raise RuntimeError(
                "runtime must be instantiated before interacting with the OMS instance"
            )
        return self._instance

    def set_result_file(self, result_file: str) -> None:
        self._require_instance().setResultFile(result_file)

    def set_logging_interval(self, interval: float) -> None:
        self._require_instance().setLoggingInterval(interval)

    def set_fixed_step_size(self, step_size: float) -> None:
        self._require_instance().setFixedStepSize(step_size)

    def initialize(self) -> None:
        self._require_instance().initialize()

    def terminate(self) -> None:
        if self._instance is None:
            return
        try:
            self._instance.terminate()
        finally:
            delete = getattr(self._instance, "delete", None)
            if callable(delete):
                try:
                    delete()
                except Exception:
                    pass
            self._instance = None

    def step_until(self, target_time: float) -> None:
        self._require_instance().stepUntil(target_time)

    def get_value(
        self, system_name: str, component_name: str, variable_name: str
    ) -> Any:
        return self._require_instance().getValue(
            self.cref(system_name, component_name, variable_name)
        )

    def set_value(
        self, system_name: str, component_name: str, variable_name: str, value: Any
    ) -> None:
        self._require_instance().setValue(
            self.cref(system_name, component_name, variable_name), value
        )

    def get_vector(
        self,
        system_name: str,
        component_name: str,
        signal_name: str,
        width: int = 5,
    ) -> list[Any]:
        return [
            self.get_value(system_name, component_name, f"{signal_name}[{index}]")
            for index in range(1, width + 1)
        ]

    def set_vector(
        self,
        system_name: str,
        component_name: str,
        signal_name: str,
        values: list[Any],
    ) -> None:
        for index, value in enumerate(values, start=1):
            self.set_value(
                system_name, component_name, f"{signal_name}[{index}]", value
            )

    def read_first_available_vector(
        self,
        system_name: str,
        component_name: str,
        signal_candidates: list[str],
        width: int = 5,
    ) -> tuple[str, list[Any]]:
        last_error = None
        for signal_name in signal_candidates:
            try:
                return signal_name, self.get_vector(
                    system_name, component_name, signal_name, width=width
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised by unit test via fake runtime
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"No signal candidates provided for {component_name}")

    def get_values(
        self,
        bindings: Mapping[str, tuple[str, str]],
        system_name: str = "default",
    ) -> dict[str, Any]:
        return {
            alias: self.get_value(system_name, component_name, variable_name)
            for alias, (component_name, variable_name) in bindings.items()
        }

    def get_binding_value(
        self,
        binding: SignalBinding,
        system_name: str = "default",
    ) -> Any:
        if binding.is_vector:
            return self.get_vector(
                system_name,
                binding.component_name,
                binding.signal_name,
                width=binding.width,
            )
        return self.get_value(system_name, binding.component_name, binding.signal_name)

    def get_bound_values(
        self,
        bindings: Mapping[str, SignalBinding],
        system_name: str = "default",
    ) -> dict[str, Any]:
        return {
            alias: self.get_binding_value(binding, system_name=system_name)
            for alias, binding in bindings.items()
        }

    def set_values(
        self,
        values: Mapping[str, Any],
        bindings: Mapping[str, tuple[str, str]],
        system_name: str = "default",
    ) -> None:
        for alias, value in values.items():
            component_name, variable_name = bindings[alias]
            self.set_value(system_name, component_name, variable_name, value)

    def set_binding_value(
        self,
        binding: SignalBinding,
        value: Any,
        system_name: str = "default",
    ) -> None:
        if binding.is_vector:
            if not isinstance(value, list):
                raise TypeError("vector signal binding requires list value")
            if len(value) != binding.width:
                raise ValueError(
                    f"vector signal binding expects {binding.width} values, got {len(value)}"
                )
            self.set_vector(
                system_name,
                binding.component_name,
                binding.signal_name,
                value,
            )
            return
        self.set_value(system_name, binding.component_name, binding.signal_name, value)
