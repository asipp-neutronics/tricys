from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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
        base_dir: Optional[str] = None,
    ) -> WorkflowPipeline:
        """Main builder entrypoint."""
        # Check if user explicitly defined a pipeline in config
        if "pipeline" in config and isinstance(config["pipeline"], dict):
            return cls._build_from_explicit_pipeline_config(
                config["pipeline"], config, export_csv, base_dir=base_dir
            )

        # Otherwise, auto-infer from standard tricys configuration
        return cls._build_by_inference(config, export_csv, base_dir=base_dir)

    @classmethod
    def _build_from_explicit_pipeline_config(
        cls,
        pipeline_cfg: Dict[str, Any],
        full_config: Dict[str, Any],
        export_csv: bool,
        base_dir: Optional[str] = None,
    ) -> WorkflowPipeline:
        # Preprocessing plugins
        preprocessors = []
        for p_spec in pipeline_cfg.get("preprocessing", []):
            preprocessors.append(
                PluginRegistry.resolve_preprocess_plugin(p_spec, base_dir=base_dir)
            )

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
            postprocessors.append(
                PluginRegistry.resolve_postprocess_plugin(p_spec, base_dir=base_dir)
            )

        # Apply stable DAG topological sort on plugins based on declarative I/O contracts
        sorted_preprocessors = cls.sort_plugins_by_dependency(preprocessors)
        sorted_postprocessors = cls.sort_plugins_by_dependency(postprocessors)

        return WorkflowPipeline(
            engine=engine,
            preprocessors=sorted_preprocessors,
            postprocessors=sorted_postprocessors,
            name=pipeline_cfg.get("name", "ConfiguredPipeline"),
        )

    @classmethod
    def sort_plugins_by_dependency(cls, plugins: list) -> list:
        """Applies stable DAG topological sorting based on requires/produces declarations."""
        if len(plugins) <= 1:
            return list(plugins)

        # Helper to extract requirements and outputs
        def get_requires(p: Any) -> set:
            reqs = set(getattr(p, "requires", []) or [])
            if hasattr(p, "get_metadata"):
                try:
                    meta = p.get_metadata()
                    reqs.update(getattr(meta, "input_requirements", []) or [])
                except Exception:
                    pass
            return {str(r).lower() for r in reqs}

        def get_produces(p: Any) -> set:
            prods = set(getattr(p, "produces", []) or [])
            name = getattr(p, "name", "")
            if name:
                prods.add(name)
            if name in ("storage", "hdf5_storage", "simulation_storage"):
                prods.update(["results_store_path", "hdf5", "simulation_results"])
            if hasattr(p, "get_metadata"):
                try:
                    meta = p.get_metadata()
                    prods.update(getattr(meta, "output_artifacts", []) or [])
                except Exception:
                    pass
            return {str(pr).lower() for pr in prods}

        # Build graph
        n = len(plugins)
        in_degree = [0] * n
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(n):
                if i != j:
                    p_i_prods = get_produces(plugins[i])
                    p_j_reqs = get_requires(plugins[j])
                    # If plugin j requires something produced by plugin i, i must precede j (i -> j)
                    if p_i_prods.intersection(p_j_reqs):
                        adj[i].append(j)
                        in_degree[j] += 1

        # Kahn's algorithm preserving original indices for stable sort
        queue = [i for i in range(n) if in_degree[i] == 0]
        sorted_indices = []

        while queue:
            # Pop smallest original index to maintain stability
            curr = queue.pop(0)
            sorted_indices.append(curr)

            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If all resolved without cycle, return reordered plugins
        if len(sorted_indices) == n:
            if sorted_indices != list(range(n)):
                logger.info(
                    f"Topological DAG auto-reordered plugins from {[getattr(p, 'name', '') for p in plugins]} "
                    f"to {[getattr(plugins[idx], 'name', '') for idx in sorted_indices]}"
                )
            return [plugins[idx] for idx in sorted_indices]

        # In case of unresolved cycles, fall back to original declaration
        logger.warning(
            "Cyclic or unresolved dependency detected in plugins, falling back to original order"
        )
        return list(plugins)

    @classmethod
    def _build_by_inference(
        cls,
        config: Dict[str, Any],
        export_csv: bool,
        base_dir: Optional[str] = None,
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
        report_cfg = (
            config.get("visualizer", {}).get("report")
            if isinstance(config.get("visualizer"), dict)
            else None
        ) or config.get("report")
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
        auditor_cfg = (
            config.get("task", {}).get("auditor")
            if isinstance(config.get("task"), dict)
            else None
        ) or config.get("auditor", {})
        auditor_enabled = bool(auditor_cfg.get("enabled", False))
        alarm_rules = config.get("static_alarm", {}).get("rules", [])
        if auditor_enabled or alarm_rules:
            safety_cls = PluginRegistry.get_postprocess_class("safety")
            if safety_cls:
                postprocessors.append(
                    safety_cls(
                        auditor_config=auditor_cfg,
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
                plugin = PluginRegistry.resolve_preprocess_plugin(
                    p_spec, base_dir=base_dir
                )
                if plugin:
                    preprocessors.append(plugin)
            except Exception as e:
                logger.warning(
                    f"Could not load explicit preprocess plugin: {p_spec}. {e}"
                )

        for p_spec in explicit_plugins.get("postprocess", []):
            try:
                plugin = PluginRegistry.resolve_postprocess_plugin(
                    p_spec, base_dir=base_dir
                )
                if plugin:
                    postprocessors.append(plugin)
            except Exception as e:
                logger.warning(
                    f"Could not load explicit postprocess plugin: {p_spec}. {e}"
                )

        # Apply stable DAG topological sort on plugins based on declarative I/O contracts
        sorted_preprocessors = cls.sort_plugins_by_dependency(preprocessors)
        sorted_postprocessors = cls.sort_plugins_by_dependency(postprocessors)

        return WorkflowPipeline(
            engine=engine,
            preprocessors=sorted_preprocessors,
            postprocessors=sorted_postprocessors,
            name="InferredPipeline",
        )

    @classmethod
    def compile_v2_to_explicit_pipeline(
        cls,
        config: Dict[str, Any],
        pipeline_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compiles Schema 2.0 top-level domain configuration (model, scenario, task, visualizer, plugins)
        losslessly into a pure explicit pipeline configuration where all plugins receive explicit parameters.
        """
        model_cfg = config.get("model", {})
        scenario_cfg = config.get("scenario", {})
        task_cfg = config.get("task", {})
        vis_cfg = config.get("visualizer", {})
        plugins_cfg = config.get("plugins", {})
        workspace_cfg = config.get("workspace", {})

        is_co_sim = "co_simulation" in config and isinstance(
            config.get("co_simulation"), dict
        )
        co_sim_engine_type = config.get("co_simulation", {}).get("engine")
        is_parallel = bool(
            task_cfg.get("parallel", False)
            or config.get("simulation", {}).get("concurrent", False)
        )

        preprocessing = []

        # 1. Setup Preprocessor with explicit parameters
        setup_params: Dict[str, Any] = {}
        if model_cfg.get("fileName"):
            setup_params["package_path"] = model_cfg["fileName"]
        if model_cfg.get("modelName"):
            setup_params["model_name"] = model_cfg["modelName"]
        if "parameters" in task_cfg:
            setup_params["parameters"] = task_cfg["parameters"]
        if "type" in task_cfg:
            setup_params["task_type"] = task_cfg["type"]
        preprocessing.append({"name": "setup", "params": setup_params})

        # 2. FOC pulse injection
        foc_cfg = config.get("foc") or scenario_cfg.get("pulse")
        if foc_cfg and (foc_cfg.get("foc_file") or foc_cfg.get("foc_path")):
            preprocessing.append({"name": "foc", "params": {"foc_config": foc_cfg}})

        # 3. Interceptor for offline co-simulation
        if is_co_sim and co_sim_engine_type != "online_oms":
            handlers = config.get("co_simulation", {}).get("handlers", [])
            preprocessing.append(
                {"name": "interceptor", "params": {"handlers": handlers}}
            )

        # 4. Modelica compiler for parallel execution
        if is_parallel and not is_co_sim:
            compiler_params: Dict[str, Any] = {}
            if model_cfg.get("fileName"):
                compiler_params["package_path"] = model_cfg["fileName"]
            if model_cfg.get("modelName"):
                compiler_params["model_name"] = model_cfg["modelName"]
            if model_cfg.get("cflags"):
                compiler_params["cflags"] = model_cfg["cflags"]
            if model_cfg.get("target"):
                compiler_params["target"] = model_cfg["target"]
            preprocessing.append({"name": "compiler", "params": compiler_params})

        # 5. Append plugins.preprocess
        if isinstance(plugins_cfg, dict):
            for p_spec in plugins_cfg.get("preprocess", []):
                preprocessing.append(p_spec)

        # 6. Simulation engine with explicit parameters
        if is_co_sim:
            engine_name = (
                "online_oms" if co_sim_engine_type == "online_oms" else "offline_cosim"
            )
            engine_spec: Dict[str, Any] = {"name": engine_name, "params": {}}
        else:
            sim_params: Dict[str, Any] = {}
            if "stopTime" in scenario_cfg:
                sim_params["stop_time"] = float(scenario_cfg["stopTime"])
            if "stepSize" in scenario_cfg:
                sim_params["step_size"] = float(scenario_cfg["stepSize"])
            if "startTime" in scenario_cfg:
                sim_params["start_time"] = float(scenario_cfg["startTime"])
            if "method" in scenario_cfg:
                sim_params["solver"] = scenario_cfg["method"]
            if "tolerance" in scenario_cfg:
                sim_params["tolerance"] = float(scenario_cfg["tolerance"])
            if "outputFormat" in scenario_cfg:
                sim_params["output_format"] = scenario_cfg["outputFormat"]
            if "variableFilter" in scenario_cfg:
                sim_params["variable_filter"] = scenario_cfg["variableFilter"]
            if "parallel" in task_cfg:
                sim_params["parallel"] = bool(task_cfg["parallel"])
            if "max_workers" in task_cfg:
                sim_params["max_workers"] = task_cfg["max_workers"]
            engine_spec = {"name": "standard", "params": sim_params}

        # 7. Postprocessing plugins
        postprocessing = []

        # 7.1 Storage plugin with explicit parameters
        storage_params: Dict[str, Any] = {}
        task_metrics = task_cfg.get("metrics") or task_cfg.get("metrics_definition")
        if task_metrics:
            storage_params["metrics_definition"] = task_metrics
        if workspace_cfg.get("results_dir"):
            storage_params["results_dir"] = workspace_cfg["results_dir"]
        postprocessing.append({"name": "storage", "params": storage_params})

        # 7.2 Safety auditor
        auditor_cfg = task_cfg.get("auditor") or config.get("auditor")
        alarm_rules = config.get("static_alarm", {}).get("rules", [])
        if auditor_cfg or alarm_rules:
            safety_params: Dict[str, Any] = {}
            if auditor_cfg:
                safety_params["auditor_config"] = auditor_cfg
            if alarm_rules:
                safety_params["alarm_rules"] = alarm_rules
            postprocessing.append({"name": "safety", "params": safety_params})

        # 7.3 Sensitivity analysis
        sensitivity_cfg = config.get("sensitivity_analysis") or (
            task_cfg if task_cfg.get("type") == "sensitivity" else None
        )
        if sensitivity_cfg and (
            sensitivity_cfg.get("enabled", False)
            or task_cfg.get("type") == "sensitivity"
        ):
            postprocessing.append(
                {
                    "name": "sensitivity_analysis",
                    "params": {"enabled": True, "sensitivity_config": sensitivity_cfg},
                }
            )

        # 7.4 Append plugins.postprocess
        if isinstance(plugins_cfg, dict):
            for p_spec in plugins_cfg.get("postprocess", []):
                postprocessing.append(p_spec)

        # 7.5 Report delivery
        report_cfg = (
            vis_cfg.get("report") if isinstance(vis_cfg, dict) else config.get("report")
        )
        if report_cfg:
            postprocessing.append(
                {"name": "report", "params": {"report_config": report_cfg}}
            )

        return {
            "pipeline": {
                "name": pipeline_name
                or config.get("pipeline", {}).get("name")
                or "CompiledExplicitPipeline",
                "preprocessing": preprocessing,
                "simulation": engine_spec,
                "postprocessing": postprocessing,
            }
        }


def compile_v2_to_explicit_pipeline(
    config: Dict[str, Any], pipeline_name: Optional[str] = None
) -> Dict[str, Any]:
    """Module-level convenience wrapper for PipelineBuilder.compile_v2_to_explicit_pipeline."""
    return PipelineBuilder.compile_v2_to_explicit_pipeline(
        config, pipeline_name=pipeline_name
    )
