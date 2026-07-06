from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import logging
import os
from typing import Iterable, Sequence

import pandas as pd

from tricys.auditor.offline import parse_auditor_config
from tricys.auditor.online import OnlineGlobalAuditor
from tricys.core.modelica import get_om_session, load_modelica_package
from tricys.online_cosim.oms_runtime import OmsSystemRuntime
from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.recorder import InMemoryStepRecorder
from tricys.online_cosim.schema import (
    OmsStepDescriptor,
    SignalBinding,
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)
from tricys.online_cosim.topology import (
    ModelConnection,
    add_topology_connection,
    build_component_type_map,
    extract_model_topology,
)

logger = logging.getLogger(__name__)


class OnlineCosimulationRunner:
    """Lightweight step dispatcher for online co-simulation processors.

    Phase 1 intentionally keeps the runner independent from OMSimulator. It only
    standardizes processor lifecycle and per-step dispatch so later phases can
    plug in OMS-backed state collection and result recording.
    """

    def __init__(
        self,
        processors: Sequence[AbstractTrackProcessor],
        recorder: object | None = None,
        auditor: OnlineGlobalAuditor | None = None,
    ) -> None:
        self._processors = list(processors)
        self._recorder = recorder
        self._auditor = auditor
        self._initialized = False
        self._contexts: list[TrackProcessorContext] = []

    @property
    def contexts(self) -> tuple[TrackProcessorContext, ...]:
        return tuple(self._contexts)

    @property
    def processors(self) -> tuple[AbstractTrackProcessor, ...]:
        return tuple(self._processors)

    def initialize(
        self,
        contexts: Sequence[TrackProcessorContext] | None = None,
    ) -> None:
        if self._initialized:
            return

        if contexts is not None and len(contexts) != len(self._processors):
            raise ValueError("contexts length must match processors length")

        self._contexts = (
            list(contexts)
            if contexts is not None
            else [
                TrackProcessorContext(
                    processor_name=processor.__class__.__name__,
                    processor_index=index,
                )
                for index, processor in enumerate(self._processors)
            ]
        )

        initialized_processors: list[AbstractTrackProcessor] = []
        try:
            for processor, context in zip(self._processors, self._contexts):
                processor.initialize(context)
                initialized_processors.append(processor)
        except Exception:
            for processor in reversed(initialized_processors):
                processor.finalize()
            self._contexts = []
            raise

        self._initialized = True

    def run_steps(
        self,
        requests: Iterable[UnifiedStateVector],
    ) -> list[list[TrackResult]]:
        if not self._initialized:
            raise RuntimeError("runner must be initialized before run_steps")

        all_results: list[list[TrackResult]] = []
        for request in requests:
            processor_results = [
                self._ensure_track_result(processor.process(request))
                for processor in self._processors
            ]
            step_results = list(processor_results)
            if self._recorder is not None:
                record_step = getattr(self._recorder, "record_step", None)
                if callable(record_step):
                    record_step(request, step_results)
            all_results.append(step_results)
        return all_results

    def run_oms_steps(
        self,
        oms_runtime: OmsSystemRuntime,
        steps: Iterable[OmsStepDescriptor],
        system_name: str = "default",
    ) -> list[list[TrackResult]]:
        if not self._initialized:
            raise RuntimeError("runner must be initialized before run_oms_steps")
        all_results: list[list[TrackResult]] = []

        current_batch: list[OmsStepDescriptor] = []
        current_key: tuple[int, int, float, float] | None = None

        def flush_batch(batch: list[OmsStepDescriptor]) -> None:
            if not batch:
                return

            batch_records: list[
                tuple[OmsStepDescriptor, UnifiedStateVector, TrackResult]
            ] = []
            for step in batch:
                processor_index = step.processor_index
                if processor_index >= len(self._processors):
                    raise IndexError(
                        f"processor_index {processor_index} is out of range for {len(self._processors)} processors"
                    )

                if self._contexts:
                    context_config = self._contexts[processor_index].config
                    context_config.setdefault(
                        "output_names", list(step.output_bindings.keys())
                    )
                    context_config.setdefault(
                        "output_bindings",
                        _serialize_signal_bindings(step.output_bindings),
                    )

                global_mass_error = 0.0
                if self._auditor is not None:
                    global_mass_error = self._auditor.state.mass_error

                extra_state_vals = oms_runtime.get_bound_values(
                    step.extra_state_bindings,
                    system_name=system_name,
                )
                # Convert from list values to simple floats for extra_state (since width=1 is assumed)
                extra_state_dict = {
                    k: v[0] if isinstance(v, list) and v else v
                    for k, v in extra_state_vals.items()
                }

                request = UnifiedStateVector(
                    component_name=step.component_name,
                    step_id=step.step_id,
                    seq_id=step.seq_id,
                    current_time_h=step.current_time_h,
                    dt_slow_h=step.dt_slow_h,
                    boundary_inputs=oms_runtime.get_bound_values(
                        step.input_bindings,
                        system_name=system_name,
                    ),
                    extra_state=extra_state_dict,
                    global_mass_error=global_mass_error,
                )
                processor_result = self._ensure_track_result(
                    self._processors[processor_index].process(request)
                )
                batch_records.append((step, request, processor_result))

            for step, _, result in batch_records:
                if getattr(result, "fallback_to_fmu", False):
                    continue
                for output_name, binding in step.output_bindings.items():
                    if output_name not in result.outputs:
                        raise KeyError(
                            f"processor result missing output '{output_name}' for component {step.component_name}"
                        )

                    output_value = result.outputs[output_name]
                    oms_runtime.set_binding_value(
                        binding,
                        self._normalize_binding_output(binding, output_value),
                        system_name=system_name,
                    )

            oms_runtime.step_until(batch[0].target_time_h)

            if self._auditor is not None:
                processor_inventory = sum(
                    p.get_mass_inventory() for p in self._processors
                )

                processor_decay_rate = 0.0
                for processor in self._processors:
                    if hasattr(processor, "get_decay_rate"):
                        processor_decay_rate += float(processor.get_decay_rate())

                self._auditor.execute_audit_step(
                    oms_runtime,
                    batch[0].dt_slow_h,
                    system_name=system_name,
                    processor_inventory=processor_inventory,
                    processor_decay_rate=processor_decay_rate,
                )

            step_results: list[TrackResult] = []
            for _, request, result in batch_records:
                if self._recorder is not None:
                    record_step = getattr(self._recorder, "record_step", None)
                    if callable(record_step):
                        record_step(request, [result])
                step_results.append(result)

            all_results.append(step_results)

        logger = logging.getLogger(__name__)
        total_steps = len(steps) if hasattr(steps, '__len__') else "unknown"
        logger.info(f"Starting online cosimulation with {total_steps} steps.")

        for i, step in enumerate(steps, 1):
            if i % 10 == 0 or i == 1:
                logger.info(f"Processing co-simulation step {i}/{total_steps} (Time: {step.current_time_h:.3f}h -> {step.target_time_h:.3f}h)")

            batch_key = (
                step.step_id,
                step.seq_id,
                step.current_time_h,
                step.target_time_h,
            )
            if current_key is None or batch_key == current_key:
                current_batch.append(step)
                current_key = batch_key
                continue

            flush_batch(current_batch)
            current_batch = [step]
            current_key = batch_key

        flush_batch(current_batch)

        return all_results

    @staticmethod
    def _ensure_track_result(result: TrackResult) -> TrackResult:
        if not isinstance(result, TrackResult):
            raise TypeError("processor must return a TrackResult instance")
        return result

    @staticmethod
    def _normalize_binding_output(binding, value):
        if binding.is_vector:
            if not isinstance(value, list):
                value = [value]
            if len(value) == 1 and binding.width > 1:
                value = value * binding.width
            if len(value) != binding.width:
                raise ValueError(
                    f"vector binding expects {binding.width} values, got {len(value)}"
                )
            return value

        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(
                    "scalar binding output list must contain exactly one value"
                )
            return value[0]

        return value

    def finalize(self) -> None:
        if not self._initialized:
            return

        try:
            for processor in reversed(self._processors):
                processor.finalize()
        finally:
            self._contexts = []
            self._initialized = False


def parse_signal_binding(binding_value) -> SignalBinding:
    if isinstance(binding_value, SignalBinding):
        return binding_value

    if isinstance(binding_value, str):
        if "." not in binding_value:
            raise ValueError(
                f"Signal binding string '{binding_value}' must use 'component.signal' format"
            )
        component_name, signal_name = binding_value.split(".", 1)
        return SignalBinding(component_name=component_name, signal_name=signal_name)

    if isinstance(binding_value, dict):
        component_name = binding_value.get("component_name")
        signal_name = binding_value.get("signal_name")
        width = binding_value.get("width", 1)
        if not isinstance(component_name, str) or not isinstance(signal_name, str):
            raise ValueError(
                "Signal binding dict requires string component_name and signal_name"
            )
        return SignalBinding(
            component_name=component_name, signal_name=signal_name, width=int(width)
        )

    raise TypeError(f"Unsupported signal binding value: {binding_value!r}")


def _serialize_signal_bindings(
    bindings: dict[str, SignalBinding],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "component_name": binding.component_name,
            "signal_name": binding.signal_name,
            "width": binding.width,
        }
        for name, binding in bindings.items()
    }


def load_handler_symbol(handler_config: dict):
    if handler_config.get("handler_script_path"):
        script_path = os.path.abspath(handler_config["handler_script_path"])
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load handler script: {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(handler_config["handler_module"])

    return getattr(module, handler_config["handler_function"])


def build_online_processor(handler_config: dict) -> AbstractTrackProcessor:
    symbol = load_handler_symbol(handler_config)
    params = dict(handler_config.get("params", {}))

    if isinstance(symbol, type) and issubclass(symbol, AbstractTrackProcessor):
        processor = symbol(**params)
    else:
        processor = symbol(**params)

    if not isinstance(processor, AbstractTrackProcessor):
        raise TypeError(
            "online_oms handler factory must return an AbstractTrackProcessor instance"
        )

    fmu_fallback = handler_config.get("shadow_fmu_fallback", {})
    if fmu_fallback.get("enabled"):
        import logging

        from tricys.online_cosim.processor_wrappers import ShadowFMUFallbackWrapper

        logging.getLogger(__name__).info(
            f"Enabling shadow FMU fallback for {handler_config.get('instance_name', 'processor')}"
        )
        processor = ShadowFMUFallbackWrapper(delegate=processor)

    time_scale = handler_config.get("time_scale_coordination", {})
    mode = time_scale.get("mode", "delegate")
    component_step_size_h = float(time_scale.get("component_step_size_h", 0.0))
    if mode == "runtime_coordinated" and component_step_size_h > 0:
        import logging

        from tricys.online_cosim.processor_wrappers import (
            RuntimeTimeScaleCoordinatorWrapper,
        )

        logging.getLogger(__name__).info(
            f"Enabling runtime time scale coordinator wrapper for {handler_config.get('instance_name', 'processor')} (step_size: {component_step_size_h}h)"
        )
        processor = RuntimeTimeScaleCoordinatorWrapper(
            delegate=processor, component_step_size_h=component_step_size_h
        )

    mass_control = handler_config.get("local_mass_control", {})
    if mass_control.get("enabled"):
        deadband_g = float(mass_control.get("deadband_g", 0.1))
        tau = float(mass_control.get("relaxation_time_h", 1.0))
        import logging

        from tricys.online_cosim.processor_wrappers import LocalMassCompensatorWrapper

        logging.getLogger(__name__).info(
            f"Enabling local mass control wrapper for {handler_config.get('instance_name', 'processor')}"
        )
        processor = LocalMassCompensatorWrapper(
            delegate=processor, deadband_g=deadband_g, relaxation_time_h=tau
        )

    return processor


def build_online_context(
    handler_config: dict,
    processor: AbstractTrackProcessor,
    *,
    processor_index: int = 0,
    global_config: dict | None = None,
) -> TrackProcessorContext:
    context_config = dict(handler_config.get("params", {}))

    # Inject shadow_fmu_path if enabled
    fmu_fallback = handler_config.get("shadow_fmu_fallback", {})
    if fmu_fallback.get("enabled"):
        fmu_path = fmu_fallback.get("fmu_path")
        if not fmu_path and global_config:
            submodel_name = handler_config.get("submodel_name")
            if submodel_name:
                import os

                short_name = submodel_name.split(".")[-1]

                package_path = global_config.get("paths", {}).get("package_path", "")
                if package_path.lower().endswith(".ssp"):
                    import shutil
                    import zipfile

                    extract_dir = os.path.join(
                        global_config.get("paths", {}).get("temp_dir", "temp"),
                        "ssp_fmus",
                    )
                    os.makedirs(extract_dir, exist_ok=True)
                    fmu_path = os.path.join(extract_dir, f"{short_name}.fmu").replace(
                        "\\", "/"
                    )
                    if not os.path.exists(fmu_path):
                        try:
                            with zipfile.ZipFile(package_path, "r") as z:
                                fmu_zip_path = f"resources/{short_name}.fmu"
                                if fmu_zip_path in z.namelist():
                                    source = z.open(fmu_zip_path)
                                    with open(fmu_path, "wb") as target:
                                        shutil.copyfileobj(source, target)
                        except Exception as e:
                            import logging

                            logging.getLogger(__name__).warning(
                                f"Could not extract {short_name}.fmu from SSP: {e}"
                            )
                            fmu_path = None
                else:
                    fmu_dir = global_config.get("co_simulation", {}).get("fmu_dir")
                    if fmu_dir:
                        fmu_path = os.path.join(fmu_dir, f"{short_name}.fmu").replace(
                            "\\", "/"
                        )

        context_config["shadow_fmu_path"] = fmu_path

    output_names = handler_config.get("output_names")
    if output_names is not None:
        context_config["output_names"] = list(output_names)
        output_bindings = handler_config.get("output_bindings")
        if isinstance(output_bindings, dict):
            context_config["output_bindings"] = _serialize_signal_bindings(
                output_bindings
            )
    return TrackProcessorContext(
        processor_name=processor.__class__.__name__,
        processor_index=processor_index,
        config=context_config,
    )


def _group_track_connections(
    connections: list[ModelConnection],
    instance_name: str,
) -> tuple[dict[str, ModelConnection], dict[str, ModelConnection]]:
    input_connections = {}
    output_connections = {}
    for connection in connections:
        source_component, source_signal = connection.source.split(".", 1)
        target_component, target_signal = connection.target.split(".", 1)
        if target_component == instance_name:
            input_connections[target_signal] = connection
        if source_component == instance_name:
            output_connections[source_signal] = connection
    return input_connections, output_connections


def _infer_track_signal_dimensions(
    omc, submodel_name: str
) -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    components = omc.sendExpression(f"getComponents({submodel_name})") or []
    input_dims = {}
    output_dims = {}
    input_order = []
    output_order = []
    for component in components:
        component_type = component[0]
        component_name = component[1]
        dimension = (
            int(component[11][0]) if len(component) > 11 and component[11] else 1
        )
        if component_type == "Modelica.Blocks.Interfaces.RealInput":
            input_dims[component_name] = dimension
            input_order.append(component_name)
        elif component_type == "Modelica.Blocks.Interfaces.RealOutput":
            output_dims[component_name] = dimension
            output_order.append(component_name)
    return input_dims, output_dims, input_order, output_order


def infer_handler_bindings(
    config: dict,
    handler_config: dict | None = None,
    *,
    omc=None,
) -> tuple[dict[str, SignalBinding], dict[str, SignalBinding], str]:
    co_simulation = config["co_simulation"]
    handler = handler_config or co_simulation["handlers"][0]
    model_name = config["simulation"]["model_name"]
    instance_name = handler["instance_name"]
    submodel_name = handler["submodel_name"]

    package_path = os.path.abspath(config["paths"]["package_path"])
    is_ssp = package_path.lower().endswith(".ssp")

    if is_ssp:
        import re
        import xml.etree.ElementTree as ET
        import zipfile

        from tricys.online_cosim.schema import SignalBinding

        with zipfile.ZipFile(package_path) as z:
            ssd_content = z.read("SystemStructure.ssd")

        root = ET.fromstring(ssd_content)

        input_dims = {}
        output_dims = {}
        input_order = []
        output_order = []

        for comp in root.iter():
            if comp.tag.endswith("Component") and comp.get("name") == instance_name:
                for connector in comp.iter():
                    if connector.tag.endswith("Connector"):
                        kind = connector.get("kind")
                        name = connector.get("name")
                        match = re.match(r"^(.*?)\[(\d+)\]$", name)
                        if match:
                            base_name = match.group(1)
                            idx = int(match.group(2))
                        else:
                            base_name = name
                            idx = 1

                        if kind == "input":
                            if base_name not in input_order:
                                input_order.append(base_name)
                            input_dims[base_name] = max(
                                input_dims.get(base_name, 0), idx
                            )
                        elif kind == "output":
                            if base_name not in output_order:
                                output_order.append(base_name)
                            output_dims[base_name] = max(
                                output_dims.get(base_name, 0), idx
                            )
                break

        input_bindings = {}
        output_bindings = {}

        for conn in root.iter():
            if conn.tag.endswith("Connection"):
                start_elem = conn.get("startElement")
                start_conn = conn.get("startConnector")
                end_elem = conn.get("endElement")
                end_conn = conn.get("endConnector")

                start_base = re.sub(r"\[\d+\]$", "", start_conn) if start_conn else ""
                end_base = re.sub(r"\[\d+\]$", "", end_conn) if end_conn else ""

                if end_elem == instance_name:
                    input_bindings[end_base] = SignalBinding(
                        component_name=start_elem,
                        signal_name=start_base,
                        width=input_dims.get(end_base, 1),
                    )
                if start_elem == instance_name:
                    output_bindings[start_base] = SignalBinding(
                        component_name=end_elem,
                        signal_name=end_base,
                        width=output_dims.get(start_base, 1),
                    )

        input_bindings = {
            k: input_bindings[k] for k in input_order if k in input_bindings
        }
        output_bindings = {
            k: output_bindings[k] for k in output_order if k in output_bindings
        }

        return input_bindings, output_bindings, "default"

    created_omc = False
    if omc is None:
        omc = get_om_session()
        created_omc = True

    try:
        if created_omc:
            if not load_modelica_package(omc, package_path):
                raise RuntimeError(
                    f"Failed to load package for online OMS bindings: {package_path}"
                )

        _, connections = extract_model_topology(omc, model_name)
        input_dims, output_dims, input_order, output_order = (
            _infer_track_signal_dimensions(omc, submodel_name)
        )
        input_connections, output_connections = _group_track_connections(
            connections, instance_name
        )

        input_bindings = {}
        for signal_name in input_order:
            if signal_name not in input_connections:
                continue
            connection = input_connections[signal_name]
            source_component, source_signal = connection.source.split(".", 1)
            input_bindings[signal_name] = SignalBinding(
                component_name=source_component,
                signal_name=source_signal,
                width=input_dims.get(signal_name, 1),
            )

        resolved_output_bindings = {}
        for signal_name in output_order:
            if signal_name not in output_connections:
                continue
            connection = output_connections[signal_name]
            target_component, target_signal = connection.target.split(".", 1)
            resolved_output_bindings[signal_name] = SignalBinding(
                component_name=target_component,
                signal_name=target_signal,
                width=output_dims.get(signal_name, 1),
            )
        return input_bindings, resolved_output_bindings, instance_name
    finally:
        if created_omc:
            try:
                omc.sendExpression("quit()")
            except Exception:
                pass


def build_online_oms_steps(config: dict, *, omc=None) -> list[OmsStepDescriptor]:
    co_simulation = config["co_simulation"]
    simulation = config.get("simulation", {})
    handlers = list(co_simulation.get("handlers", []))
    handler_bindings = [
        infer_handler_bindings(config, handler_config=handler, omc=omc)
        for handler in handlers
    ]

    step_size = float(simulation.get("step_size", 1.0))
    start_time = float(simulation.get("start_time", 0.0))
    stop_time = float(simulation.get("stop_time", 2.0))

    steps = []
    current_time = start_time
    step_id = 0
    while current_time < stop_time - 1e-12:
        step_id += 1
        target_time = min(current_time + step_size, stop_time)
        for processor_index, (
            (input_bindings, output_bindings, component_name),
            handler_cfg,
        ) in enumerate(zip(handler_bindings, handlers)):
            extra_state_cfg = handler_cfg.get("extra_state", {})
            extra_state_bindings = {}
            for k, v in extra_state_cfg.items():
                target_component, target_signal = v.split(".", 1)
                extra_state_bindings[k] = SignalBinding(
                    component_name=target_component,
                    signal_name=target_signal,
                    width=1,
                )

            steps.append(
                OmsStepDescriptor(
                    component_name=component_name,
                    step_id=step_id,
                    seq_id=step_id,
                    current_time_h=current_time,
                    dt_slow_h=target_time - current_time,
                    target_time_h=target_time,
                    input_bindings=input_bindings,
                    output_bindings=output_bindings,
                    extra_state_bindings=extra_state_bindings,
                    processor_index=processor_index,
                )
            )
        current_time = target_time
    return steps


def resolve_online_fmu_dirs(config: dict) -> tuple[str, str]:
    co_simulation = config.get("co_simulation", {})
    explicit_fmu_dir = co_simulation.get("fmu_dir")
    if explicit_fmu_dir:
        fmu_dir = os.path.abspath(explicit_fmu_dir)
        return fmu_dir, os.path.join(fmu_dir, "temp")

    run_base_dir = os.path.dirname(config["paths"].get("temp_dir", ""))
    if run_base_dir:
        return os.path.join(run_base_dir, "fmu"), os.path.join(run_base_dir, "fmu_temp")

    package_path = os.path.abspath(config["paths"]["package_path"])
    package_dir = (
        package_path if os.path.isdir(package_path) else os.path.dirname(package_path)
    )
    fmu_dir = os.path.join(package_dir, "fmus")
    return fmu_dir, os.path.join(fmu_dir, "temp")


def resolve_online_temp_base_dir(config: dict) -> str:
    explicit_temp_dir = config.get("paths", {}).get("temp_dir")
    if explicit_temp_dir:
        return os.path.abspath(explicit_temp_dir)

    results_dir = config.get("paths", {}).get("results_dir")
    if results_dir:
        return os.path.join(os.path.dirname(os.path.abspath(results_dir)), "temp")

    return os.path.abspath("temp")


def resolve_online_result_file(config: dict) -> str:
    co_simulation = config.get("co_simulation", {})
    simulation = config.get("simulation", {})
    result_file = str(
        co_simulation.get(
            "result_file", simulation.get("result_file", "online_oms_result.csv")
        )
    )

    base, ext = os.path.splitext(result_file)
    if ext.lower() != ".csv":
        result_file = base + ".csv"

    if os.path.isabs(result_file):
        return result_file

    if "result_file" not in co_simulation and "result_file" not in simulation:
        base_temp_dir = resolve_online_temp_base_dir(config)
        return os.path.join(base_temp_dir, "job_1", result_file)

    return os.path.join(config["paths"]["results_dir"], result_file)


def _apply_online_job_parameters(
    oms_runtime: OmsSystemRuntime,
    job_params: dict,
    *,
    system_name: str,
    track_component_names: set[str],
) -> None:
    import logging

    logger = logging.getLogger(__name__)

    for parameter_name, parameter_value in job_params.items():
        if "." not in parameter_name:
            logger.debug(
                f"Skipping top-level parameter '{parameter_name}': online_oms requires 'component.parameter' format."
            )
            continue

        component_name, variable_name = parameter_name.split(".", 1)
        if component_name in track_component_names:
            logger.warning(
                f"Skipping parameter '{parameter_name}': targets externally managed component '{component_name}'."
            )
            continue

        if isinstance(parameter_value, str):
            logger.warning(
                f"Skipping parameter '{parameter_name}': string expressions ('{parameter_value}') are not supported by online_oms at runtime."
            )
            continue

        try:
            oms_runtime.set_value(
                system_name, component_name, variable_name, parameter_value
            )
        except Exception as e:
            logger.warning(f"Failed to set parameter '{parameter_name}': {e}")


def build_online_oms_runtime(
    config: dict,
    *,
    oms_runtime: OmsSystemRuntime | None = None,
    omc=None,
    job_params: dict | None = None,
) -> OmsSystemRuntime:
    package_path = os.path.abspath(config["paths"]["package_path"])
    is_ssp = package_path.lower().endswith(".ssp")
    model_name = config["simulation"]["model_name"]
    system_name = config.get("co_simulation", {}).get("system_name", "default")

    track_component_names = {
        handler["instance_name"]
        for handler in config.get("co_simulation", {}).get("handlers", [])
    }

    created_omc = False
    try:
        if is_ssp:
            from OMSimulator import SSP

            runtime = oms_runtime or OmsSystemRuntime(model=SSP(package_path))
            for comp in track_component_names:
                try:
                    runtime.model.delete(runtime.cref(system_name, comp))
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Failed to delete {comp} from SSP: {e}"
                    )
        else:
            fmu_dir, fmu_temp_dir = resolve_online_fmu_dirs(config)
            if omc is None:
                omc = get_om_session()
                created_omc = True

            if not load_modelica_package(omc, package_path):
                raise RuntimeError(
                    f"Failed to load package for online OMS assembly: {package_path}"
                )

            components, connections = extract_model_topology(omc, model_name)
            component_types = build_component_type_map(components)

            runtime = oms_runtime or OmsSystemRuntime()
            runtime.topology_components = components

            os.makedirs(fmu_dir, exist_ok=True)

            for component in components:
                fmu_path = os.path.join(fmu_dir, f"{component.class_name}.fmu")
                if not os.path.exists(fmu_path):
                    import logging

                    logging.getLogger(__name__).info(
                        f"Auto-building FMU for {component.full_class_name}..."
                    )
                    os.makedirs(fmu_temp_dir, exist_ok=True)
                    omc_cwd = omc.sendExpression("cd()")
                    try:
                        omc_temp_dir = fmu_temp_dir.replace("\\", "/")
                        omc.sendExpression(f'cd("{omc_temp_dir}")')
                        generated_fmu = omc.sendExpression(
                            f'buildModelFMU({component.full_class_name}, version="2.0", fmuType="cs")'
                        )
                        if generated_fmu:
                            fmu_out_path = (
                                generated_fmu
                                if os.path.isabs(generated_fmu)
                                else os.path.join(fmu_temp_dir, generated_fmu)
                            )
                            if os.path.exists(fmu_out_path):
                                import shutil

                                shutil.move(fmu_out_path, fmu_path)
                            else:
                                raise RuntimeError(
                                    f"Failed to auto-build FMU for {component.full_class_name}: file {fmu_out_path} not found"
                                )
                        else:
                            raise RuntimeError(
                                f"Failed to auto-build FMU for {component.full_class_name}"
                            )
                    finally:
                        if omc_cwd:
                            omc.sendExpression(f'cd("{omc_cwd.replace(chr(92), "/")}")')

                if component.instance_name in track_component_names:
                    continue

                resource_name = f"resources/{component.class_name}.fmu"
                runtime.add_resource(fmu_path.replace("\\", "/"), resource_name)
                runtime.add_component(
                    system_name, component.instance_name, resource_name
                )

            for connection in connections:
                source_component = connection.source.split(".", 1)[0]
                target_component = connection.target.split(".", 1)[0]
                if (
                    source_component in track_component_names
                    or target_component in track_component_names
                ):
                    continue
                add_topology_connection(
                    runtime,
                    component_types,
                    connection.source,
                    connection.target,
                    system_name=system_name,
                )

            export_ssp_dir = config.get("co_simulation", {}).get("export_ssp_dir")
            if not export_ssp_dir:
                results_dir = config.get("paths", {}).get("results_dir")
                if results_dir:
                    export_ssp_dir = os.path.join(
                        os.path.dirname(os.path.abspath(results_dir)), "ssp"
                    )
                else:
                    export_ssp_dir = os.path.abspath("ssp")

            abs_export_dir = os.path.abspath(export_ssp_dir)
            os.makedirs(abs_export_dir, exist_ok=True)

            package_path = config["paths"]["package_path"]
            ssp_filename = os.path.splitext(os.path.basename(package_path))[0] + ".ssp"
            abs_export_path = os.path.join(abs_export_dir, ssp_filename)

            runtime.model.export(abs_export_path)
            import logging

            logging.getLogger(__name__).info(
                f"Exported underlying SSP to {abs_export_path}"
            )

        runtime.instantiate()
        result_file = resolve_online_result_file(config)
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        runtime.set_result_file(result_file)

        step_size = float(config.get("simulation", {}).get("step_size", 1.0))
        runtime.set_logging_interval(step_size)

        if job_params:
            _apply_online_job_parameters(
                runtime,
                job_params,
                system_name=system_name,
                track_component_names=track_component_names,
            )

        runtime.initialize()
        return runtime
    finally:
        if created_omc and omc is not None:
            try:
                omc.sendExpression("quit()")
            except Exception:
                pass


def _prepare_online_job_config(config: dict, job_id: int) -> dict:
    job_config = copy.deepcopy(config)
    job_config["simulation_parameters"] = {}
    job_config["_internal_job_id"] = job_id

    job_workspace = os.path.join(
        resolve_online_temp_base_dir(config),
        f"job_{job_id}",
    )
    os.makedirs(job_workspace, exist_ok=True)

    job_config["paths"]["temp_dir"] = job_workspace
    job_config["co_simulation"] = dict(job_config["co_simulation"])
    job_config["co_simulation"]["fmu_dir"] = os.path.join(job_workspace, "fmu")
    job_config["co_simulation"]["result_file"] = os.path.join(
        job_workspace,
        "online_oms_result.csv",
    )
    return job_config


def _export_online_trace(
    recorder: object | None, trace_dir: str
) -> dict[str, object] | None:
    export_recorder_csv = getattr(recorder, "export_csv", None)
    if not callable(export_recorder_csv):
        return None
    return export_recorder_csv(trace_dir)


def _write_online_hdf_result(
    config: dict,
    result_file: str,
    params: dict,
    *,
    export_csv: bool = False,
) -> str:
    from tricys.simulation.simulation import _process_h5_result, export_results_to_csv
    from tricys.utils.file_utils import get_unique_filename

    results_dir = os.path.abspath(config["paths"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    hdf_path = get_unique_filename(results_dir, "sweep_results.h5")

    with pd.HDFStore(hdf_path, mode="w", complib="blosc", complevel=9) as store:
        try:
            config_df = pd.DataFrame({"config_json": [json.dumps(config)]})
            config_df = config_df.astype(object)
            store.put("config", config_df, format="fixed")
        except Exception:
            pass

        _process_h5_result(
            store,
            1,
            params,
            result_file,
            config,
            config.get("metrics_definition", {}),
            filter_schema=config.get("filter_schema"),
        )

    if export_csv:
        export_results_to_csv(results_dir, hdf_path)

    return hdf_path


def _run_online_cosimulation_sweep(
    config: dict,
    *,
    omc=None,
    export_csv: bool = False,
) -> dict:
    from tricys.core.jobs import generate_simulation_jobs
    from tricys.simulation.simulation import _process_h5_result, export_results_to_csv
    from tricys.utils.file_utils import get_unique_filename

    jobs = generate_simulation_jobs(config.get("simulation_parameters", {}))
    results_dir = os.path.abspath(config["paths"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    hdf_path = get_unique_filename(results_dir, "sweep_results.h5")
    job_results = []

    with pd.HDFStore(hdf_path, mode="w", complib="blosc", complevel=9) as store:
        try:
            config_df = pd.DataFrame({"config_json": [json.dumps(config)]})
            config_df = config_df.astype(object)
            store.put("config", config_df, format="fixed")
        except Exception:
            pass

        for index, job_params in enumerate(jobs, start=1):
            logger.info(
                "Running online OMS sweep job",
                extra={"job_id": index, "job_params": job_params},
            )
            job_config = _prepare_online_job_config(config, index)
            result = run_online_cosimulation(
                job_config,
                omc=omc,
                job_params=job_params,
            )
            result_file = result["result_file"]
            _process_h5_result(store, index, job_params, result_file, job_config)
            job_results.append(
                {
                    "job_id": index,
                    "job_params": job_params,
                    "result_file": result_file,
                    "online_trace": result.get("recorder_csv"),
                }
            )

    if export_csv:
        export_results_to_csv(results_dir, hdf_path)

    return {
        "hdf_path": hdf_path,
        "jobs": job_results,
        "job_count": len(job_results),
    }


def run_online_cosimulation(
    config: dict,
    *,
    oms_runtime: OmsSystemRuntime | None = None,
    omc=None,
    recorder: object | None = None,
    job_params: dict | None = None,
    export_csv: bool = False,
) -> dict:
    """Run the minimal config-driven online OMS co-simulation path.

    Phase 6 only wires the formal config entry point. Automatic OMS model
    assembly is still injected from the outside until a later phase.
    """

    simulation_parameters = config.get("simulation_parameters") or {}
    if job_params is None and simulation_parameters:
        if oms_runtime is not None:
            raise ValueError(
                "online_oms parameter sweep does not support injected oms_runtime"
            )
        return _run_online_cosimulation_sweep(
            config,
            omc=omc,
            export_csv=export_csv,
        )

    created_oms_runtime = oms_runtime is None
    if oms_runtime is None:
        oms_runtime = build_online_oms_runtime(config, omc=omc, job_params=job_params)

    auditor_config = parse_auditor_config(config)
    auditor = None
    if auditor_config.enabled:
        auditor = OnlineGlobalAuditor(auditor_config)

    handler_configs = list(config["co_simulation"].get("handlers", []))
    if not handler_configs:
        system_name = config.get("co_simulation", {}).get("system_name", "default")
        if auditor is not None:
            fmu_dir, _ = resolve_online_fmu_dirs(config)
            package_path = os.path.abspath(
                config.get("paths", {}).get("package_path", "")
            )
            auditor.initialize(
                oms_runtime,
                fmu_dir,
                getattr(oms_runtime, "topology_components", []),
                system_name=system_name,
                initial_processor_inventory=0.0,
                package_path=package_path,
            )

        step_size = float(config.get("simulation", {}).get("step_size", 1.0))
        start_time = float(config.get("simulation", {}).get("start_time", 0.0))
        stop_time = float(config.get("simulation", {}).get("stop_time", 2.0))
        current_time = start_time
        while current_time < stop_time - 1e-12:
            target_time = min(current_time + step_size, stop_time)
            dt_slow_h = target_time - current_time
            oms_runtime.step_until(target_time)
            if auditor is not None:
                auditor.execute_audit_step(
                    oms_runtime,
                    dt_slow_h,
                    system_name=system_name,
                    processor_inventory=0.0,
                    processor_decay_rate=0.0,
                )
            current_time = target_time

        results = []
        steps = []
        runner = None
    else:
        processors = [
            build_online_processor(handler_config) for handler_config in handler_configs
        ]
        runner = OnlineCosimulationRunner(
            processors,
            recorder=recorder or InMemoryStepRecorder(),
            auditor=auditor,
        )
        steps = build_online_oms_steps(config, omc=omc)
        contexts = []
        for processor_index, (handler_config, processor) in enumerate(
            zip(handler_configs, processors)
        ):
            enriched_handler_config = dict(handler_config)
            step = next(
                (
                    candidate
                    for candidate in steps
                    if candidate.processor_index == processor_index
                ),
                None,
            )
            if step is not None:
                enriched_handler_config["output_names"] = list(
                    step.output_bindings.keys()
                )
                enriched_handler_config["output_bindings"] = dict(step.output_bindings)
            contexts.append(
                build_online_context(
                    enriched_handler_config,
                    processor,
                    processor_index=processor_index,
                    global_config=config,
                )
            )

        runner.initialize(contexts=contexts)

        system_name = config.get("co_simulation", {}).get("system_name", "default")
        if auditor is not None:
            fmu_dir, _ = resolve_online_fmu_dirs(config)
            initial_processor_inventory = sum(
                p.get_mass_inventory() for p in processors
            )
            package_path = os.path.abspath(
                config.get("paths", {}).get("package_path", "")
            )

            # Exclude internal components that are overridden by Python handlers
            ignored_instances = [
                h.get("instance_name") for h in handler_configs if "instance_name" in h
            ]

            auditor.initialize(
                oms_runtime,
                fmu_dir,
                getattr(oms_runtime, "topology_components", []),
                system_name=system_name,
                initial_processor_inventory=initial_processor_inventory,
                package_path=package_path,
                ignored_instances=ignored_instances,
            )

        try:
            results = runner.run_oms_steps(
                oms_runtime,
                steps,
                system_name=config.get("co_simulation", {}).get(
                    "system_name", "default"
                ),
            )
        finally:
            runner.finalize()

    # Apply variable filter to output CSV
    result_file = resolve_online_result_file(config)

    if os.path.exists(result_file):
        import logging

        import pandas as pd

        try:
            pd.read_csv(result_file)

        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to process or filter result file {result_file}: {e}"
            )

    base_temp_dir = resolve_online_temp_base_dir(config)
    trace_dir = os.path.join(base_temp_dir, "job_1")
    if job_params is not None:
        trace_dir = base_temp_dir
    recorder_csv = _export_online_trace(
        runner._recorder if runner is not None else None, trace_dir
    )

    hdf_path = None
    if job_params is None:
        hdf_path = _write_online_hdf_result(
            config,
            result_file,
            {},
            export_csv=export_csv,
        )

    if created_oms_runtime:
        try:
            oms_runtime.terminate()
        except Exception as exc:
            logger.warning(
                "Failed to terminate OMS runtime cleanly",
                extra={"error": str(exc)},
            )

    return {
        "results": results,
        "recorder": runner._recorder if runner is not None else None,
        "steps": steps,
        "oms_runtime": oms_runtime,
        "result_file": result_file,
        "recorder_csv": recorder_csv,
        "hdf_path": hdf_path,
    }
