from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tricys.core.modelica import get_om_session, load_modelica_package
from tricys.workflow.base import PreprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_preprocess_plugin

logger = logging.getLogger(__name__)


@register_preprocess_plugin("interceptor")
@register_preprocess_plugin("interceptor_builder")
class InterceptorBuilderPlugin(PreprocessPlugin):
    """Prepares metadata, asset paths, and port mapping for offline co-simulation interceptors."""

    name = "interceptor"

    def __init__(
        self,
        handlers: Optional[List[Dict[str, Any]]] = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self.handlers = handlers

    def _resolve_handlers(self, context: PipelineContext) -> List[Dict[str, Any]]:
        if self.handlers is not None:
            return copy.deepcopy(self.handlers)
        co_sim_cfg = context.config.get("co_simulation", {})
        handlers = copy.deepcopy(co_sim_cfg.get("handlers", []))
        if not isinstance(handlers, list):
            handlers = [handlers] if handlers else []
        return handlers

    def validate(self, context: PipelineContext) -> bool:
        handlers = self._resolve_handlers(context)
        return bool(handlers)

    def run(self, context: PipelineContext) -> None:
        handlers = self._resolve_handlers(context)

        # Scan input ports for each submodel
        omc = get_om_session()
        try:
            if not load_modelica_package(omc, Path(context.package_path).as_posix()):
                raise RuntimeError(
                    f"Failed to load Modelica package at: {context.package_path}"
                )

            all_input_vars: List[str] = []
            handler_port_info: Dict[str, Any] = {}

            for handler in handlers:
                submodel_name = handler["submodel_name"]
                instance_name = handler["instance_name"]

                components = omc.sendExpression(f"getComponents({submodel_name})")
                input_ports = [
                    {"name": c[1], "dim": int(c[11][0]) if c[11] else 1}
                    for c in (components or [])
                    if c[0] == "Modelica.Blocks.Interfaces.RealInput"
                ]

                handler_port_info[instance_name] = input_ports

                for port in input_ports:
                    full_name = f"{instance_name}.{port['name']}".replace(".", "\\.")
                    if port["dim"] > 1:
                        full_name += f"\\[[1-{port['dim']}]\\]"
                    all_input_vars.append(full_name)

            context.interception_info = {
                "handlers": handlers,
                "handler_port_info": handler_port_info,
                "primary_variable_filter": (
                    "time|" + "|".join(all_input_vars) if all_input_vars else "time"
                ),
            }
            logger.info(f"Prepared interceptor metadata for {len(handlers)} handler(s)")
        finally:
            omc.sendExpression("quit()")
