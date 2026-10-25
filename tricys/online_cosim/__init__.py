"""Online stepwise co-simulation primitives.

This package hosts the new OMSimulator-oriented online co-simulation runtime.
Phase 1 only provides stable protocol objects and a lightweight processor
runner that can be validated without OMSimulator or any external solver.
"""

from tricys.online_cosim.oms_runtime import (
    OmsBindings,
    OmsBindingsError,
    OmsSystemRuntime,
)
from tricys.online_cosim.processor_base import (
    AbstractTrackProcessor,
)
from tricys.online_cosim.recorder import InMemoryStepRecorder
from tricys.online_cosim.runtime import (
    OnlineCosimulationRunner,
    build_online_oms_runtime,
    resolve_online_fmu_dirs,
    resolve_online_result_file,
    run_online_cosimulation,
)
from tricys.online_cosim.schema import (
    OmsStepDescriptor,
    SignalBinding,
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)
from tricys.online_cosim.topology import (
    ModelComponent,
    ModelConnection,
    add_topology_connection,
    build_component_type_map,
    extract_model_topology,
)

__all__ = [
    "AbstractTrackProcessor",
    "ModelComponent",
    "ModelConnection",
    "InMemoryStepRecorder",
    "OnlineCosimulationRunner",
    "build_online_oms_runtime",
    "resolve_online_fmu_dirs",
    "resolve_online_result_file",
    "OmsBindings",
    "OmsStepDescriptor",
    "OmsBindingsError",
    "OmsSystemRuntime",
    "SignalBinding",
    "TrackProcessorContext",
    "TrackResult",
    "UnifiedStateVector",
    "add_topology_connection",
    "build_component_type_map",
    "extract_model_topology",
    "run_online_cosimulation",
]
