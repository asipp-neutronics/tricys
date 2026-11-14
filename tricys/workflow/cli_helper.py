from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml

from tricys.workflow.base import BasePlugin, PostprocessPlugin, PreprocessPlugin
from tricys.workflow.builder import compile_v2_to_explicit_pipeline
from tricys.workflow.registry import PluginRegistry
from tricys.workflow.testing import create_mock_context

logger = logging.getLogger(__name__)


def generate_plugin_scaffold(
    name: str,
    category: str = "postprocess",
    output_path: Optional[str] = None,
) -> str:
    """Generates an Agent-Native 3.0 production-grade plugin template."""
    clean_name = name.strip().lower().replace("-", "_")
    class_name = "".join(part.capitalize() for part in clean_name.split("_")) + "Plugin"

    base_class = "PostprocessPlugin"
    register_decorator = "@register_postprocess_plugin"
    if category == "preprocess":
        base_class = "PreprocessPlugin"
        register_decorator = "@register_preprocess_plugin"

    code_template = f'''"""TRICYS Custom Plugin: {class_name}

Generated automatically by TRICYS Agent-Native Scaffolding CLI.
Adheres to Schema 2.0 Explicit-First Contract with Physics Semantic Bus integration.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from tricys.workflow.base import PluginMetadata, {base_class}
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import {register_decorator.strip("@")}

logger = logging.getLogger(__name__)


{register_decorator}("{clean_name}")
class {class_name}({base_class}):
    """Agent-designed custom plugin for TRICYS.

    Demonstrates:
    1. Explicit-First parameter resolution with fallback to context.
    2. Physics Semantic Bus resolution (decoupled from Modelica internal hierarchy).
    3. Outputting metrics to context.metrics and saving structured artifacts.
    """

    name = "{clean_name}"

    def __init__(
        self,
        threshold: float = 100.0,
        enable_audit: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.threshold = float(threshold)
        self.enable_audit = bool(enable_audit)

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name=cls.name,
            category="{category}",
            description="Agent-designed {category} plugin for physics evaluation.",
            version="1.0.0",
            author="TRICYS Autonomous AI Agent",
            parameters_schema={{
                "threshold": {{
                    "type": "number",
                    "default": 100.0,
                    "description": "Critical monitoring threshold value.",
                }},
                "enable_audit": {{
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to perform comprehensive validation.",
                }},
            }},
            input_requirements=["results"],
            output_artifacts=["{clean_name}_report.json"],
        )

    def validate(self, context: PipelineContext) -> bool:
        """Validates prerequisites before execution."""
        if not context.results and not context.results_store_path:
            logger.warning(f"[{{self.name}}] No in-memory simulation results or store path found.")
            return False
        return True

    def run(self, context: PipelineContext) -> None:
        """Executes the plugin logic across simulation jobs."""
        logger.info(f"[{{self.name}}] Running analysis with threshold={{self.threshold}}...")

        results_list: List[Dict[str, Any]] = []

        # Iterate over typed job results
        for job in context.results:
            df = job.dataframe
            if df is None or df.empty:
                continue

            # Query fusion timeseries via Physics Semantic Bus
            sds_inv = context.resolve_physics_variable(df, "sds_inventory")
            blanket_tbr = context.resolve_physics_variable(df, "blanket_tbr")

            min_sds = float(sds_inv.min()) if sds_inv is not None else 0.0
            avg_tbr = float(blanket_tbr.mean()) if blanket_tbr is not None else 0.0

            # Store metrics into job result and global context
            job.metrics["min_sds_inventory"] = min_sds
            job.metrics["avg_blanket_tbr"] = avg_tbr

            results_list.append({{
                "job_id": job.job_id,
                "min_sds_inventory": min_sds,
                "avg_blanket_tbr": avg_tbr,
                "status": "PASS" if min_sds >= self.threshold else "WARNING",
            }})

        # Record findings in context metadata and metrics
        context.metadata["{clean_name}_results"] = results_list
        context.metrics["{clean_name}_job_count"] = len(results_list)

        # Output artifact
        out_dir = context.results_dir or "./results"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "{clean_name}_report.json")
        import json
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({{"plugin": self.name, "results": results_list}}, f, indent=2)

        context.artifacts["{clean_name}_report"] = out_file
        logger.info(f"[{{self.name}}] Completed! Report saved to {{out_file}}")
'''

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(code_template, encoding="utf-8")
        logger.info(f"Plugin scaffold saved to {output_path}")

    return code_template


def test_plugin_in_mock_sandbox(
    target: str,
    class_name: Optional[str] = None,
    mock_duration_hours: float = 72.0,
) -> Dict[str, Any]:
    """Tests a plugin in an isolated in-memory mock context in milliseconds."""
    plugin_cls: Optional[Type[BasePlugin]] = None

    # 1. Try loading from file path
    target_path = Path(target)
    if target_path.exists() and target_path.is_file():
        spec = importlib.util.spec_from_file_location(
            f"_test_mod_{target_path.stem}", str(target_path.resolve())
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)

            # Discover plugin class
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    inspect.isclass(attr)
                    and issubclass(attr, BasePlugin)
                    and attr not in (BasePlugin, PostprocessPlugin, PreprocessPlugin)
                ):
                    if class_name is None or attr_name == class_name:
                        plugin_cls = attr
                        break
    else:
        # 2. Try looking up in registry
        plugin_cls = PluginRegistry.get_plugin(target)

    if not plugin_cls:
        raise ValueError(
            f"Could not locate a valid BasePlugin class in '{target}' (class_name={class_name})"
        )

    # Instantiate plugin
    plugin_inst = plugin_cls()

    # Create high-fidelity in-memory mock context
    mock_context = create_mock_context(
        num_jobs=1,
        stop_time=mock_duration_hours * 3600.0,
        step_size=360.0,
    )

    # Execute lifecycle
    is_valid = plugin_inst.validate(mock_context)
    if not is_valid:
        return {
            "status": "VALIDATION_FAILED",
            "plugin": plugin_cls.__name__,
            "message": "Plugin validate() returned False against mock context.",
        }

    plugin_inst.run(mock_context)

    return {
        "status": "SUCCESS",
        "plugin": plugin_cls.__name__,
        "metrics_emitted": mock_context.metrics,
        "metadata_emitted": list(mock_context.metadata.keys()),
        "artifacts_emitted": mock_context.artifacts,
        "results_count": len(mock_context.results),
    }


def compile_v2_cli(
    config_path: str,
    output_path: Optional[str] = None,
    as_json: bool = False,
) -> str:
    """CLI handler for compiling Schema 2.0 into explicit pipeline structure."""
    cfg_p = Path(config_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    text = cfg_p.read_text(encoding="utf-8")
    if cfg_p.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    compiled = compile_v2_to_explicit_pipeline(data)

    if as_json:
        formatted_str = json.dumps(compiled, indent=2, ensure_ascii=False)
    else:
        formatted_str = yaml.dump(
            compiled, sort_keys=False, allow_unicode=True, indent=2
        )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(formatted_str, encoding="utf-8")
        logger.info(f"Compiled explicit pipeline saved to {output_path}")

    return formatted_str
