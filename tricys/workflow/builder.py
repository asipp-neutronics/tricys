from __future__ import annotations

import logging
from typing import Any, Dict

from tricys.workflow.base import PipelineError
from tricys.workflow.pipeline import WorkflowPipeline
from tricys.workflow.registry import PluginRegistry

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Constructs a WorkflowPipeline either from explicit configuration or by auto-inference."""

    @classmethod
    def build(
        cls,
        config: Dict[str, Any],
        export_csv: bool = False,
    ) -> WorkflowPipeline:
        """Main builder entrypoint."""
        # Check if user explicitly defined a pipeline in config
        if "pipeline" in config and isinstance(config["pipeline"], dict):
            return cls._build_from_explicit_pipeline_config(
                config["pipeline"], config, export_csv
            )

        # Otherwise, auto-infer from standard tricys configuration
        return cls._build_by_inference(config, export_csv)

    @classmethod
    def _build_from_explicit_pipeline_config(
        cls,
        pipeline_cfg: Dict[str, Any],
        full_config: Dict[str, Any],
        export_csv: bool,
    ) -> WorkflowPipeline:
        # Preprocessing plugins
        preprocessors = []
        for p_spec in pipeline_cfg.get("preprocessing", []):
            preprocessors.append(PluginRegistry.resolve_preprocess_plugin(p_spec))

        # Simulation Engine
        engine_spec = pipeline_cfg.get("simulation", {})
        if isinstance(engine_spec, str):
            engine_cls = PluginRegistry.get_engine_class(engine_spec)
            if not engine_cls:
                raise PipelineError(
                    f"Simulation engine '{engine_spec}' not found in registry"
                )
            engine = engine_cls()
        elif isinstance(engine_spec, dict):
            engine_name = engine_spec.get("name") or engine_spec.get("type", "standard")
            engine_cls = PluginRegistry.get_engine_class(engine_name)
            if not engine_cls:
                raise PipelineError(
                    f"Simulation engine '{engine_name}' not found in registry"
                )
            engine_params = engine_spec.get("params", {})
            engine = engine_cls(**engine_params)
        else:
            raise PipelineError(
                "Explicit pipeline must specify a valid 'simulation' engine block"
            )

        # Postprocessing plugins
        postprocessors = []
        for p_spec in pipeline_cfg.get("postprocessing", []):
            postprocessors.append(PluginRegistry.resolve_postprocess_plugin(p_spec))

        return WorkflowPipeline(
            engine=engine,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            name=pipeline_cfg.get("name", "ConfiguredPipeline"),
        )

    @classmethod
    def _build_by_inference(
        cls,
        config: Dict[str, Any],
        export_csv: bool,
    ) -> WorkflowPipeline:
        """Intelligently infer the required plugins based on standard config sections."""
        preprocessors = []
        postprocessors = []

        is_co_sim = "co_simulation" in config and isinstance(
            config.get("co_simulation"), dict
        )
        co_sim_engine_type = config.get("co_simulation", {}).get("engine")

        # ----------------------------------------------------
        # 1. Inferred Preprocessing (Functional capability plugins)
        # ----------------------------------------------------
        # 1.1 Environment setup: Workspace isolation & parameter sweep job expansion
        setup_cls = PluginRegistry.get_preprocess_class("setup")
        if setup_cls:
            preprocessors.append(setup_cls())

        # 1.2 FOC pulse scenario injection (if configured)
        foc_cfg = config.get("foc", {})
        if foc_cfg and foc_cfg.get("foc_path"):
            foc_cls = PluginRegistry.get_preprocess_class("foc")
            if foc_cls:
                preprocessors.append(foc_cls(foc_config=foc_cfg))

        # 1.3 Co-simulation interceptor interface configuration (if offline co-sim)
        if is_co_sim and co_sim_engine_type != "online_oms":
            interceptor_cls = PluginRegistry.get_preprocess_class("interceptor")
            if interceptor_cls:
                handlers = config.get("co_simulation", {}).get("handlers", [])
                preprocessors.append(interceptor_cls(handlers=handlers))

        # 1.4 Modelica static compilation for concurrent runs
        sim_cfg = config.get("simulation", {})
        task_cfg = config.get("task", {})
        is_parallel = sim_cfg.get("concurrent", False) or (
            isinstance(task_cfg, dict) and task_cfg.get("parallel", False)
        )
        if is_parallel and not is_co_sim:
            compiler_cls = PluginRegistry.get_preprocess_class("compiler")
            if compiler_cls:
                preprocessors.append(compiler_cls())

        # ----------------------------------------------------
        # 2. Inferred Simulation Engine
        # ----------------------------------------------------
        if is_co_sim:
            if co_sim_engine_type == "online_oms":
                engine_cls = PluginRegistry.get_engine_class("online_oms")
            else:
                engine_cls = PluginRegistry.get_engine_class("offline_cosim")
        else:
            engine_cls = PluginRegistry.get_engine_class("standard")

        if not engine_cls:
            raise PipelineError(
                "Failed to resolve simulation engine during pipeline inference"
            )

        engine = engine_cls()

        # ----------------------------------------------------
        # 3. Inferred Postprocessing (Functional capability plugins)
        # ----------------------------------------------------
        # 3.1 Data persistence & export service (metrics + HDF5 + optional CSV)
        storage_cls = PluginRegistry.get_postprocess_class("storage")
        if storage_cls:
            task_metrics = (
                task_cfg.get("metrics") or task_cfg.get("metrics_definition")
                if isinstance(task_cfg, dict)
                else None
            )
            metrics_def = (
                config.get("metrics_definition")
                or task_metrics
                or config.get("sensitivity_analysis", {}).get("metrics_definition", {})
            )
            postprocessors.append(
                storage_cls(
                    export_csv=export_csv,
                    metrics_definition=metrics_def,
                )
            )

        # 3.2 Engineering report generation & multi-format delivery
        report_cfg = config.get("report") or config.get("visualizer", {}).get("report")
        if report_cfg or config.get("post_processing"):
            report_cls = PluginRegistry.get_postprocess_class("report")
            if report_cls:
                postprocessors.append(
                    report_cls(
                        report_config=report_cfg,
                        tasks=config.get("post_processing"),
                    )
                )

        # 3.3 Safety margin & mass balance conservation auditor
        auditor_enabled = config.get("auditor", {}).get("enabled", False)
        alarm_rules = config.get("static_alarm", {}).get("rules", [])
        if auditor_enabled or alarm_rules:
            safety_cls = PluginRegistry.get_postprocess_class("safety")
            if safety_cls:
                postprocessors.append(
                    safety_cls(
                        auditor_config=config.get("auditor", {}),
                        alarm_rules=alarm_rules,
                    )
                )

        # 3.4 Sensitivity & uncertainty analysis
        sensitivity_cfg = config.get("sensitivity_analysis", {})
        is_v2_sensitivity = (
            isinstance(task_cfg, dict) and task_cfg.get("type") == "sensitivity"
        )
        if sensitivity_cfg.get("enabled", False) or is_v2_sensitivity:
            sens_cls = PluginRegistry.get_postprocess_class("sensitivity_analysis")
            if sens_cls:
                postprocessors.append(
                    sens_cls(
                        enabled=True,
                        sensitivity_config=(
                            sensitivity_cfg if not is_v2_sensitivity else task_cfg
                        ),
                    )
                )

        # ----------------------------------------------------
        # 4. Explicit V2 Plugins Configuration
        # ----------------------------------------------------
        explicit_plugins = config.get("plugins", {})
        for p_spec in explicit_plugins.get("preprocess", []):
            try:
                plugin = PluginRegistry.resolve_preprocess_plugin(p_spec)
                if plugin:
                    preprocessors.append(plugin)
            except Exception as e:
                logger.warning(
                    f"Could not load explicit preprocess plugin: {p_spec}. {e}"
                )

        for p_spec in explicit_plugins.get("postprocess", []):
            try:
                plugin = PluginRegistry.resolve_postprocess_plugin(p_spec)
                if plugin:
                    postprocessors.append(plugin)
            except Exception as e:
                logger.warning(
                    f"Could not load explicit postprocess plugin: {p_spec}. {e}"
                )

        return WorkflowPipeline(
            engine=engine,
            preprocessors=preprocessors,
            postprocessors=postprocessors,
            name="InferredPipeline",
        )
