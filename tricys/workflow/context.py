from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from tricys.core.semantic import PhysicsSemanticResolver


@dataclass
class SimulationJobResult:
    """Strongly-typed contract for an individual simulation job execution result."""

    job_id: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    result_path: str = ""
    dataframe: Optional[pd.DataFrame] = None
    time_column: str = "time"
    time_unit: str = "second"
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Extensible task-level payload for plugins (Open-Closed Principle)
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Lazy-loads and caches pandas DataFrame from result_path if not already in memory."""
        if self.dataframe is not None:
            return self.dataframe
        if self.result_path and os.path.exists(self.result_path):
            try:
                self.dataframe = pd.read_csv(self.result_path)
                return self.dataframe
            except Exception:
                return None
        return None


@dataclass
class PipelineContext:
    """Universal state carrier flowing across Preprocessing -> Simulation -> Postprocessing."""

    # Configurations
    config: Dict[str, Any]
    original_config: Dict[str, Any] = field(default_factory=dict)
    base_dir: str = ""
    temp_dir: str = ""
    results_dir: str = ""

    # Model and Execution Specifications
    package_path: str = ""
    model_name: str = ""
    stop_time: float = 0.0
    step_size: float = 0.0
    time_base: str = "second"

    # Preprocessing Artifacts
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    compiled_artifacts: Dict[str, str] = field(default_factory=dict)  # exe, xml, om_bin
    interception_info: Dict[str, Any] = field(default_factory=dict)

    # Simulation Outputs
    results_store_path: Optional[str] = None  # Path to HDF5 results
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    results: List[SimulationJobResult] = field(default_factory=list)
    execution_status: str = "initialized"
    execution_error: Optional[str] = None

    # Physics Semantic Bus Resolver
    semantic_resolver: Optional[PhysicsSemanticResolver] = None

    # Extensible payload for plugins and inter-stage metadata (Open-Closed Principle)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_definition: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    typed_config: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.semantic_resolver is None:
            custom_map = self.config.get("semantic_mapping")
            self.semantic_resolver = PhysicsSemanticResolver(custom_mapping=custom_map)

    def sync_typed_results(self) -> List[SimulationJobResult]:
        """Synchronizes legacy raw_results list into strongly-typed SimulationJobResult objects."""
        if not self.results and self.raw_results:
            synced = []
            for item in self.raw_results:
                r_obj = SimulationJobResult(
                    job_id=int(item.get("job_id", 1)),
                    parameters=dict(item.get("params", {})),
                    result_path=str(item.get("result_path", "")),
                    dataframe=item.get("dataframe"),
                    metrics=dict(item.get("metrics", {})),
                    extra=dict(item.get("extra", {})),
                )
                synced.append(r_obj)
            self.results = synced
        return self.results

    def resolve_physics_variable(
        self,
        df: pd.DataFrame,
        concept: str,
    ) -> Optional[pd.Series]:
        """Resolves a standardized physics concept (e.g. 'sds_inventory') from timeseries dataframe."""
        if self.semantic_resolver:
            return self.semantic_resolver.resolve_variable(df, concept)
        return None

    def get_matched_column(
        self,
        columns_or_df: Any,
        concept: str,
    ) -> Optional[str]:
        """Discovers the concrete Modelica column name for a standardized physics concept."""
        if self.semantic_resolver:
            cols = (
                columns_or_df.columns
                if hasattr(columns_or_df, "columns")
                else columns_or_df
            )
            return self.semantic_resolver.get_matched_column(cols, concept)
        return None

    def get_all_inventories(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """Extracts all component inventories present in dataframe as a clean dictionary."""
        if self.semantic_resolver:
            return self.semantic_resolver.get_all_inventories(df)
        return {}

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        original_config: Optional[Dict[str, Any]] = None,
        base_dir: Optional[str] = None,
    ) -> PipelineContext:
        """Factory method to construct a PipelineContext from standard config."""
        cfg = copy.deepcopy(config)
        orig_cfg = copy.deepcopy(
            original_config if original_config is not None else config
        )

        resolved_base_dir = os.path.abspath(base_dir or os.getcwd())
        paths_cfg = cfg.get("paths", {})
        sim_cfg = cfg.get("simulation", {})

        raw_results_dir = paths_cfg.get("results_dir", "results")
        raw_temp_dir = paths_cfg.get("temp_dir", "temp")
        raw_pkg_path = paths_cfg.get("package_path", "")

        results_dir = (
            raw_results_dir
            if os.path.isabs(raw_results_dir)
            else os.path.abspath(os.path.join(resolved_base_dir, raw_results_dir))
        )
        temp_dir = (
            raw_temp_dir
            if os.path.isabs(raw_temp_dir)
            else os.path.abspath(os.path.join(resolved_base_dir, raw_temp_dir))
        )
        package_path = (
            raw_pkg_path
            if (not raw_pkg_path or os.path.isabs(raw_pkg_path))
            else os.path.abspath(os.path.join(resolved_base_dir, raw_pkg_path))
        )

        task_cfg = cfg.get("task", {})
        task_metrics = (
            task_cfg.get("metrics") or task_cfg.get("metrics_definition")
            if isinstance(task_cfg, dict)
            else None
        )
        metrics_def = (
            cfg.get("metrics_definition")
            or task_metrics
            or cfg.get("sensitivity_analysis", {}).get("metrics_definition", {})
        )

        typed_cfg = None
        try:
            from tricys.core.schema import (
                TricysConfig,
                TricysV2Config,
                is_v2_config,
            )

            if is_v2_config(cfg):
                typed_cfg = TricysV2Config.from_dict(cfg)
            else:
                typed_cfg = TricysConfig.from_dict(cfg)
        except Exception:
            pass

        return cls(
            config=cfg,
            original_config=orig_cfg,
            base_dir=resolved_base_dir,
            temp_dir=temp_dir,
            results_dir=results_dir,
            package_path=package_path,
            model_name=sim_cfg.get("model_name", ""),
            stop_time=float(sim_cfg.get("stop_time", 0.0)),
            step_size=float(sim_cfg.get("step_size", 0.0)),
            time_base=str(sim_cfg.get("time_base", "second")),
            metrics_definition=metrics_def,
            typed_config=typed_cfg,
        )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
