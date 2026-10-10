from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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

    # Preprocessing Artifacts
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    compiled_artifacts: Dict[str, str] = field(default_factory=dict)  # exe, xml, om_bin
    interception_info: Dict[str, Any] = field(default_factory=dict)

    # Simulation Outputs
    results_store_path: Optional[str] = None  # Path to HDF5 results
    raw_results: List[Dict[str, Any]] = field(default_factory=list)
    execution_status: str = "initialized"
    execution_error: Optional[str] = None

    # Extensible payload for plugins and inter-stage metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_definition: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    typed_config: Optional[Any] = None

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
            metrics_definition=metrics_def,
            typed_config=typed_cfg,
        )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value
