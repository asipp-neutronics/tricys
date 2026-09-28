from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from tricys.core.modelica import get_om_session, load_modelica_package
from tricys.workflow.base import PreprocessPlugin
from tricys.workflow.context import PipelineContext
from tricys.workflow.registry import register_preprocess_plugin

logger = logging.getLogger(__name__)


def resolve_built_model_paths(build_result: list, build_dir: str) -> Tuple[str, str]:
    executable_artifact = str(
        (build_result[0] if len(build_result) > 0 else "") or ""
    ).strip()
    xml_artifact = str((build_result[1] if len(build_result) > 1 else "") or "").strip()

    if not executable_artifact or not xml_artifact:
        raise RuntimeError(f"Model build returned invalid artifacts: {build_result!r}")

    executable_name = executable_artifact
    if sys.platform == "win32" and not executable_name.lower().endswith(".exe"):
        executable_name = f"{executable_name}.exe"

    executable_path = (
        executable_name
        if os.path.isabs(executable_name)
        else os.path.join(build_dir, executable_name)
    )
    xml_path = (
        xml_artifact
        if os.path.isabs(xml_artifact)
        else os.path.join(build_dir, xml_artifact)
    )
    return executable_path, xml_path


def build_model_artifacts(
    package_path: Union[str, Dict[str, Any], None] = None,
    model_name: Optional[str] = None,
    temp_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[str, str, str]:
    """Compiles Modelica model once into executable, xml, and discovers om_bin.

    Supports:
      - build_model_artifacts(config)
      - build_model_artifacts(package_path, model_name, temp_dir)
      - build_model_artifacts(package_path=..., model_name=..., temp_dir=...)
      - build_model_artifacts(package_path_or_config=...)
    """
    if isinstance(package_path, dict):
        cfg = package_path
    elif config is not None:
        cfg = config
    elif kwargs.get("package_path_or_config") is not None:
        arg = kwargs["package_path_or_config"]
        if isinstance(arg, dict):
            cfg = arg
        else:
            package_path = str(arg)
            cfg = None
    else:
        cfg = None

    if cfg is not None:
        pkg_p = os.path.abspath(cfg["paths"]["package_path"])
        mdl_n = cfg["simulation"]["model_name"]
        tmp_d = os.path.abspath(cfg["paths"].get("temp_dir", "temp"))
    else:
        if not package_path or not model_name or not temp_dir:
            raise ValueError(
                "package_path, model_name and temp_dir are required when passing arguments separately"
            )
        pkg_p = os.path.abspath(str(package_path))
        mdl_n = str(model_name)
        tmp_d = os.path.abspath(str(temp_dir))

    build_dir = os.path.abspath(os.path.join(tmp_d, "build"))
    os.makedirs(build_dir, exist_ok=True)
    logger.info(f"Building Modelica model '{mdl_n}' in {build_dir}...")

    omc = get_om_session()
    try:
        omc.sendExpression(f'cd("{Path(build_dir).as_posix()}")')

        if not load_modelica_package(omc, Path(pkg_p).as_posix()):
            raise RuntimeError(f"Failed to load Modelica package at: {pkg_p}")

        build_result = omc.sendExpression(f"buildModel({mdl_n})")
        if not build_result or len(build_result) < 2:
            err = omc.sendExpression("getErrorString()")
            raise RuntimeError(f"Model build failed: {err}")

        try:
            exe_path, xml_path = resolve_built_model_paths(build_result, build_dir)
        except RuntimeError as exc:
            err = omc.sendExpression("getErrorString()")
            raise RuntimeError(f"Model build failed: {err or exc}") from exc

        om_home = omc.sendExpression("getInstallationDirectoryPath()")
        om_bin_path = os.path.join(om_home, "bin")

        logger.info(f"Model built successfully: {exe_path}")
        return exe_path, xml_path, om_bin_path
    finally:
        omc.sendExpression("quit()")


@register_preprocess_plugin("compiler")
@register_preprocess_plugin("modelica_compiler")
class ModelCompilerPlugin(PreprocessPlugin):
    """Pre-compiles the Modelica model to enable fast subprocess execution."""

    name = "compiler"

    def run(self, context: PipelineContext) -> None:
        exe, xml, om_bin = build_model_artifacts(
            package_path=context.package_path,
            model_name=context.model_name,
            temp_dir=context.temp_dir,
        )
        context.compiled_artifacts = {
            "exe": exe,
            "xml": xml,
            "om_bin": om_bin,
            "build_dir": os.path.dirname(exe),
        }
