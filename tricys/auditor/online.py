import fnmatch
import logging
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from typing import Any

from tricys.auditor.offline import AuditorConfig
from tricys.online_cosim.oms_runtime import OmsSystemRuntime


@dataclass(slots=True)
class AuditorState:
    time: float = 0.0
    initial_mass: float = 0.0
    current_mass: float = 0.0
    cumulative_sources: float = 0.0
    cumulative_leak: float = 0.0
    cumulative_burn: float = 0.0
    cumulative_decay: float = 0.0
    mass_error: float = 0.0


logger = logging.getLogger(__name__)


class OnlineGlobalAuditor:
    def __init__(self, config: AuditorConfig):
        self.config = config
        self.state = AuditorState()
        self.inv_bindings: list[tuple[str, str]] = []
        self.src_bindings: list[tuple[str, str]] = []
        self.leak_bindings: list[tuple[str, str]] = []
        self.burn_bindings: list[tuple[str, str]] = []
        self.decay_bindings: list[tuple[str, str]] = []
        self.cum_src_bindings: list[tuple[str, str]] = []
        self.cum_leak_bindings: list[tuple[str, str]] = []
        self.cum_burn_bindings: list[tuple[str, str]] = []
        self.cum_decay_bindings: list[tuple[str, str]] = []
        self._initialized = False

    def initialize(
        self,
        oms_runtime,
        fmu_dir: str,
        components: list[Any],
        system_name: str = "default",
        initial_processor_inventory: float = 0.0,
        package_path: str = "",
        ignored_instances: list[str] | None = None,
    ) -> None:
        """Initialize the auditor by sniffing variables from the FMU XML directly, supporting both standalone FMUs and SSPs."""
        if not self.config.enabled:
            return

        ignored_instances = ignored_instances or []

        # Handle explicit components (from .mo compilation)
        for comp in components:
            if comp.instance_name in ignored_instances:
                continue
            fmu_path = os.path.join(fmu_dir, f"{comp.class_name}.fmu")
            variables = self._get_component_variables(fmu_path)
            self._register_variables(comp.instance_name, variables)

        # Handle SSP components directly from the zip
        if not components and package_path and package_path.lower().endswith(".ssp"):
            import zipfile
            from io import BytesIO

            try:
                with zipfile.ZipFile(package_path, "r") as ssp_zip:
                    ssd_path = "SystemStructure.ssd"
                    if ssd_path in ssp_zip.namelist():
                        with ssp_zip.open(ssd_path) as ssd_f:
                            ssd_tree = ET.parse(ssd_f)
                            ssd_root = ssd_tree.getroot()
                            for comp_elem in ssd_root.iter():
                                if comp_elem.tag.endswith("Component"):
                                    comp_name = comp_elem.get("name")
                                    source = comp_elem.get("source")
                                    if comp_name and source and source.endswith(".fmu"):
                                        if source in ssp_zip.namelist():
                                            try:
                                                fmu_bytes = ssp_zip.read(source)
                                                with zipfile.ZipFile(
                                                    BytesIO(fmu_bytes), "r"
                                                ) as fmu_zip:
                                                    if (
                                                        "modelDescription.xml"
                                                        in fmu_zip.namelist()
                                                    ):
                                                        with fmu_zip.open(
                                                            "modelDescription.xml"
                                                        ) as md_f:
                                                            md_tree = ET.parse(md_f)
                                                            md_root = md_tree.getroot()
                                                            variables = []
                                                            model_vars = md_root.find(
                                                                ".//ModelVariables"
                                                            )
                                                            if model_vars is not None:
                                                                for (
                                                                    scalar_var
                                                                ) in model_vars.findall(
                                                                    ".//ScalarVariable"
                                                                ):
                                                                    var_name = (
                                                                        scalar_var.get(
                                                                            "name"
                                                                        )
                                                                    )
                                                                    if var_name:
                                                                        variables.append(
                                                                            var_name
                                                                        )
                                                            if (
                                                                comp_name
                                                                not in ignored_instances
                                                            ):
                                                                self._register_variables(
                                                                    comp_name, variables
                                                                )
                                            except Exception as e:
                                                logger.warning(
                                                    f"Failed to read variables from {source} inside SSP: {e}"
                                                )
            except Exception as e:
                logger.warning(
                    f"Failed to extract components from SSP {package_path}: {e}"
                )

        self._finalize_initialization(
            oms_runtime, system_name, initial_processor_inventory
        )

    def _register_variables(self, instance_name: str, variables: list[str]) -> None:
        for var_name in variables:
            full_name = f"{instance_name}.{var_name}"

            if any(
                fnmatch.fnmatch(full_name, p) for p in self.config.inventory_patterns
            ):
                self.inv_bindings.append((instance_name, var_name))
            if any(fnmatch.fnmatch(full_name, p) for p in self.config.source_patterns):
                self.src_bindings.append((instance_name, var_name))
            if any(fnmatch.fnmatch(full_name, p) for p in self.config.leak_patterns):
                self.leak_bindings.append((instance_name, var_name))
            if any(fnmatch.fnmatch(full_name, p) for p in self.config.burn_patterns):
                self.burn_bindings.append((instance_name, var_name))
            if any(fnmatch.fnmatch(full_name, p) for p in self.config.decay_patterns):
                self.decay_bindings.append((instance_name, var_name))
            if any(
                fnmatch.fnmatch(full_name, p)
                for p in self.config.cumulative_source_patterns
            ):
                self.cum_src_bindings.append((instance_name, var_name))
            if any(
                fnmatch.fnmatch(full_name, p)
                for p in self.config.cumulative_leak_patterns
            ):
                self.cum_leak_bindings.append((instance_name, var_name))
            if any(
                fnmatch.fnmatch(full_name, p)
                for p in self.config.cumulative_burn_patterns
            ):
                self.cum_burn_bindings.append((instance_name, var_name))
            if any(
                fnmatch.fnmatch(full_name, p)
                for p in self.config.cumulative_decay_patterns
            ):
                self.cum_decay_bindings.append((instance_name, var_name))

    def _finalize_initialization(
        self, oms_runtime, system_name: str, initial_processor_inventory: float
    ) -> None:
        # Capture initial mass
        masses = []
        total_fmu = 0.0
        for comp_name, var_name in self.inv_bindings:
            try:
                val = float(oms_runtime.get_value(system_name, comp_name, var_name))
                total_fmu += val
                if val > 0:
                    masses.append(f"{comp_name}.{var_name}={val:.4f}")
            except Exception as e:
                logger.warning(f"Failed to read init {comp_name}.{var_name}: {e}")
        self.state.current_mass = total_fmu + initial_processor_inventory
        self.state.initial_mass = self.state.current_mass
        self.state.time = 0.0
        self._initialized = True
        logger.info(
            f"Global Auditor initialized. Initial mass: {self.state.initial_mass} (FMU: {total_fmu}, Proc: {initial_processor_inventory})"
        )
        logger.info(f"Initial FMU masses: {', '.join(masses)}")
        logger.info(
            f"Discovered bounds: inv={len(self.inv_bindings)}, src={len(self.src_bindings)}, leak={len(self.leak_bindings)}, burn={len(self.burn_bindings)}, decay={len(self.decay_bindings)}"
        )
        logger.info(
            f"Discovered cum bounds: c_src={len(self.cum_src_bindings)}, c_leak={len(self.cum_leak_bindings)}, c_burn={len(self.cum_burn_bindings)}, c_decay={len(self.cum_decay_bindings)}"
        )

    def _get_component_variables(self, fmu_path: str) -> list[str]:
        variables = []
        if not os.path.exists(fmu_path):
            return variables

        try:
            with zipfile.ZipFile(fmu_path, "r") as z:
                with z.open("modelDescription.xml") as f:
                    tree = ET.parse(f)
                    root = tree.getroot()

                    # Extract variables from FMI 2.0 structure
                    model_vars = root.find("ModelVariables")
                    if model_vars is not None:
                        for scalar_var in model_vars.findall("ScalarVariable"):
                            name = scalar_var.get("name")
                            if name:
                                variables.append(name)
        except Exception as e:
            logger.warning(f"Failed to parse variables from {fmu_path}: {e}")

        return variables

    def _sum_bindings(
        self,
        oms_runtime: OmsSystemRuntime,
        system_name: str,
        bindings: list[tuple[str, str]],
    ) -> float:
        total = 0.0
        for comp_name, var_name in bindings:
            try:
                total += float(oms_runtime.get_value(system_name, comp_name, var_name))
            except Exception as e:
                logger.warning(f"Failed to read {comp_name}.{var_name}: {e}")
        return total

    def execute_audit_step(
        self,
        oms_runtime,
        dt: float,
        system_name: str = "default",
        processor_inventory: float = 0.0,
        processor_decay_rate: float = 0.0,
    ) -> None:
        """Run the mass balance tracking for one step."""
        if not self.config.enabled:
            return

        src_rate = self._sum_bindings(oms_runtime, system_name, self.src_bindings)
        burn_rate = self._sum_bindings(oms_runtime, system_name, self.burn_bindings)
        leak_rate = self._sum_bindings(oms_runtime, system_name, self.leak_bindings)
        decay_rate = (
            self._sum_bindings(oms_runtime, system_name, self.decay_bindings)
            + processor_decay_rate
        )

        # Trapezoidal integration
        if hasattr(self, "_prev_rates"):
            p_src, p_burn, p_leak, p_decay = self._prev_rates
            self.state.cumulative_sources += (p_src + src_rate) / 2.0 * dt
            self.state.cumulative_burn += (p_burn + burn_rate) / 2.0 * dt
            self.state.cumulative_leak += (p_leak + leak_rate) / 2.0 * dt
            self.state.cumulative_decay += (p_decay + decay_rate) / 2.0 * dt
        else:
            self.state.cumulative_sources += src_rate * dt
            self.state.cumulative_burn += burn_rate * dt
            self.state.cumulative_leak += leak_rate * dt
            self.state.cumulative_decay += decay_rate * dt

        self._prev_rates = (src_rate, burn_rate, leak_rate, decay_rate)

        current_inventory = self._sum_bindings(
            oms_runtime, system_name, self.inv_bindings
        )
        self.state.current_mass = current_inventory + processor_inventory
        self.state.time += dt

        # Read exact cumulative variables directly
        cum_src_direct = self._sum_bindings(
            oms_runtime, system_name, self.cum_src_bindings
        )
        cum_burn_direct = self._sum_bindings(
            oms_runtime, system_name, self.cum_burn_bindings
        )
        cum_leak_direct = self._sum_bindings(
            oms_runtime, system_name, self.cum_leak_bindings
        )
        cum_decay_direct = self._sum_bindings(
            oms_runtime, system_name, self.cum_decay_bindings
        )

        total_sources = self.state.cumulative_sources + cum_src_direct
        total_burn = self.state.cumulative_burn + cum_burn_direct
        total_leak = self.state.cumulative_leak + cum_leak_direct
        total_decay = self.state.cumulative_decay + cum_decay_direct

        expected_mass = (
            self.state.initial_mass
            + total_sources
            - total_leak
            - total_burn
            - total_decay
        )
        self.state.mass_error = self.state.current_mass - expected_mass

        logger.info(
            f"Audit Step (dt={dt}h): TotalSources={total_sources:.3f}, CurrentMass={self.state.current_mass:.3f}, MassError={self.state.mass_error:.2e}"
        )

        if self.config.warn_threshold_g == 0:
            logger.info(
                f"Audit [t={self.state.time:.2f}]: Mass error = {self.state.mass_error:.4f} g"
            )
            logger.info(
                f"  -> Details: expected={expected_mass:.4f}, current={self.state.current_mass:.4f} | init={self.state.initial_mass:.4f}, +src={total_sources:.4f}, -leak={total_leak:.4f}, -burn={total_burn:.4f}, -decay={total_decay:.4f} | fmu={current_inventory:.4f}, proc={processor_inventory:.4f}"
            )
        elif (
            self.config.warn_threshold_g > 0
            and abs(self.state.mass_error) > self.config.warn_threshold_g
        ):
            logger.warning(
                f"Global mass balance error is drifting: {self.state.mass_error:.4f} g"
            )
            logger.warning(
                f"DEBUG: expected={expected_mass:.4f}, current={self.state.current_mass:.4f} | init={self.state.initial_mass:.4f}, +src={total_sources:.4f}, -leak={total_leak:.4f}, -burn={total_burn:.4f}, -decay={total_decay:.4f} | fmu={current_inventory:.4f}, proc={processor_inventory:.4f}"
            )

        if (
            self.config.kill_threshold_g > 0
            and abs(self.state.mass_error) > self.config.kill_threshold_g
        ):
            logger.error(
                f"Global mass balance error exceeds {self.config.kill_threshold_g:.1f} g safety threshold: {self.state.mass_error:.4f} g"
            )
            # Log exact components for debug
            logger.debug(
                f"AUDIT FAIL DETAILS: current_mass={self.state.current_mass:.4f}, expected={expected_mass:.4f}"
            )
            logger.debug(
                f"AUDIT FAIL DETAILS: FMU_inv={current_inventory:.4f}, Processor_inv={processor_inventory:.4f}"
            )
            logger.debug(
                f"AUDIT FAIL DETAILS: sum_sources={self.state.cumulative_sources:.4f}, sum_burn={self.state.cumulative_burn:.4f}"
            )
            raise RuntimeError(
                f"Global mass balance error ({self.state.mass_error:.4f} g) exceeds safety threshold of {self.config.kill_threshold_g:.1f} g. Simulation terminated."
            )
