from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from tricys.workflow.base import (
    BaseSimulationEngine,
    PipelineError,
    PostprocessPlugin,
    PreprocessPlugin,
    SecurityValidationError,
)

logger = logging.getLogger(__name__)

FORBIDDEN_MODULES = {
    "subprocess",
    "pty",
    "commands",
    "multiprocessing",
    "threading",
    "socket",
    "urllib.request",
    "http.client",
    "requests",
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
}

FORBIDDEN_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "spawn"),
    ("os", "exec"),
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("shutil", "rmtree"),
}


def validate_script_ast_safety(source_code: str, file_path: str = "") -> None:
    """Statically validates Python source code to prevent malicious system calls."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise SecurityValidationError(f"Plugin script syntax error: {e}") from e

    for node in ast.walk(tree):
        # 1. Check forbidden module imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if mod_root in FORBIDDEN_MODULES:
                    raise SecurityValidationError(
                        f"Security violation at line {node.lineno}: import of '{alias.name}' is strictly forbidden in plugin sandbox."
                    )
        elif isinstance(node, ast.ImportFrom):
            mod_root = (node.module or "").split(".")[0]
            if mod_root in FORBIDDEN_MODULES:
                raise SecurityValidationError(
                    f"Security violation at line {node.lineno}: from '{node.module}' import is strictly forbidden in plugin sandbox."
                )

        # 2. Check forbidden calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    raise SecurityValidationError(
                        f"Security violation at line {node.lineno}: call to '{node.func.id}()' is strictly forbidden in plugin sandbox."
                    )
            elif isinstance(node.func, ast.Attribute):
                attr_name = node.func.attr
                if isinstance(node.func.value, ast.Name):
                    caller = node.func.value.id
                    if (caller, attr_name) in FORBIDDEN_ATTR_CALLS or any(
                        caller == f_caller and attr_name.startswith(f_attr)
                        for f_caller, f_attr in FORBIDDEN_ATTR_CALLS
                    ):
                        raise SecurityValidationError(
                            f"Security violation at line {node.lineno}: call to '{caller}.{attr_name}()' is strictly forbidden in plugin sandbox."
                        )


class PluginRegistry:
    """Central registry and dynamic resolver for workflow plugins and simulation engines."""

    _preprocess_registry: Dict[str, Type[PreprocessPlugin]] = {}
    _postprocess_registry: Dict[str, Type[PostprocessPlugin]] = {}
    _engine_registry: Dict[str, Type[BaseSimulationEngine]] = {}

    @classmethod
    def register_preprocess(cls, name: str, plugin_cls: Type[PreprocessPlugin]) -> None:
        cls._preprocess_registry[name.lower()] = plugin_cls
        logger.debug(
            f"Registered preprocess plugin: '{name.lower()}' -> {plugin_cls.__name__}"
        )

    @classmethod
    def register_postprocess(
        cls, name: str, plugin_cls: Type[PostprocessPlugin]
    ) -> None:
        cls._postprocess_registry[name.lower()] = plugin_cls
        logger.debug(
            f"Registered postprocess plugin: '{name.lower()}' -> {plugin_cls.__name__}"
        )

    @classmethod
    def register_engine(cls, name: str, engine_cls: Type[BaseSimulationEngine]) -> None:
        cls._engine_registry[name.lower()] = engine_cls
        logger.debug(
            f"Registered simulation engine: '{name.lower()}' -> {engine_cls.__name__}"
        )

    @classmethod
    def get_preprocess_class(cls, name: str) -> Optional[Type[PreprocessPlugin]]:
        return cls._preprocess_registry.get(name.lower())

    @classmethod
    def get_postprocess_class(cls, name: str) -> Optional[Type[PostprocessPlugin]]:
        return cls._postprocess_registry.get(name.lower())

    @classmethod
    def get_engine_class(cls, name: str) -> Optional[Type[BaseSimulationEngine]]:
        return cls._engine_registry.get(name.lower())

    @classmethod
    def introspect(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Collects structured metadata for all registered plugins and engines."""
        import tricys.workflow  # noqa: F401

        categories = {
            "preprocess": cls._preprocess_registry,
            "engine": cls._engine_registry,
            "postprocess": cls._postprocess_registry,
        }

        result: Dict[str, List[Dict[str, Any]]] = {}
        for cat_name, registry in categories.items():
            seen_classes = set()
            cat_list = []
            for name, plugin_cls in registry.items():
                if plugin_cls in seen_classes:
                    continue
                seen_classes.add(plugin_cls)
                aliases = sorted([k for k, v in registry.items() if v is plugin_cls])
                if hasattr(plugin_cls, "get_metadata") and callable(
                    plugin_cls.get_metadata
                ):
                    meta = plugin_cls.get_metadata().to_dict()
                else:
                    meta = {
                        "name": name,
                        "category": cat_name,
                        "description": (plugin_cls.__doc__ or "").strip(),
                        "version": "2.0.0",
                        "author": "TRICYS Core Team",
                        "parameters_schema": {},
                        "input_requirements": [],
                        "output_artifacts": [],
                    }
                meta["aliases"] = aliases
                cat_list.append(meta)
            result[cat_name] = cat_list
        return result

    @classmethod
    def get_plugin_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific plugin by name."""
        name_lower = name.lower()
        plugin_cls = (
            cls.get_preprocess_class(name_lower)
            or cls.get_postprocess_class(name_lower)
            or cls.get_engine_class(name_lower)
        )
        if not plugin_cls:
            return None
        if hasattr(plugin_cls, "get_metadata") and callable(plugin_cls.get_metadata):
            return plugin_cls.get_metadata().to_dict()
        return {
            "name": name,
            "description": (plugin_cls.__doc__ or "").strip(),
        }

    @classmethod
    def load_class_from_source(
        cls,
        source: Union[str, Dict[str, Any]],
        expected_base: Optional[Type[Any]] = None,
        base_dir: Optional[str] = None,
    ) -> Type[Any]:
        """Dynamically load a class from a script path or python module path with AST security guard.

        Supported formats:
        - dict with 'script_path' (and optional 'class_name' or 'class')
        - dict with 'module' (and optional 'class_name' or 'class')
        - string path to a .py file
        - string module path like 'my_pkg.plugins.MyPlugin'
        """
        script_path_str: Optional[str] = None
        module_name_str: Optional[str] = None
        class_name: Optional[str] = None

        if isinstance(source, dict):
            script_path_str = source.get("script") or source.get("script_path")
            module_name_str = source.get("module")
            class_name = source.get("class") or source.get("class_name")
        elif isinstance(source, str):
            if source.endswith(".py") or "\\" in source or "/" in source:
                script_path_str = source
            else:
                module_name_str = source

        loaded_module = None
        if script_path_str:
            path = Path(script_path_str)
            if not path.is_absolute() and base_dir:
                candidate = (Path(base_dir) / path).resolve()
                if candidate.is_file():
                    path = candidate
                else:
                    path = path.resolve()
            else:
                path = path.resolve()

            if not path.is_file() and base_dir:
                candidate = (Path(base_dir) / script_path_str).resolve()
                if candidate.is_file():
                    path = candidate

            if not path.is_file():
                raise PipelineError(f"Plugin script not found at {path}")

            # AST Security Sandboxing Guard
            code_text = path.read_text(encoding="utf-8")
            validate_script_ast_safety(code_text, str(path))

            mod_name = path.stem
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if not spec or not spec.loader:
                raise PipelineError(f"Failed to create module spec for {path}")
            loaded_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded_module)
        elif module_name_str:
            if "." in module_name_str and not class_name:
                parts = module_name_str.split(".")
                if parts[-1][0].isupper():
                    class_name = parts[-1]
                    module_name_str = ".".join(parts[:-1])
            loaded_module = importlib.import_module(module_name_str)
        else:
            raise PipelineError(f"Invalid plugin source specification: {source}")

        if class_name:
            if not hasattr(loaded_module, class_name):
                raise PipelineError(
                    f"Class '{class_name}' not found in module {loaded_module.__name__}"
                )
            target_cls = getattr(loaded_module, class_name)
        else:
            # Auto-discover first class inheriting from expected_base
            candidates = []
            for _, obj in inspect.getmembers(loaded_module, inspect.isclass):
                if (
                    expected_base
                    and issubclass(obj, expected_base)
                    and obj is not expected_base
                ):
                    candidates.append(obj)
            if not candidates:
                raise PipelineError(
                    f"No subclass of {expected_base.__name__ if expected_base else 'expected class'} found in {loaded_module}"
                )
            target_cls = candidates[0]

        if expected_base and not issubclass(target_cls, expected_base):
            raise PipelineError(
                f"Loaded class {target_cls} is not a subclass of {expected_base.__name__}"
            )

        return target_cls

    @classmethod
    def resolve_preprocess_plugin(
        cls,
        spec: Union[str, Dict[str, Any], PreprocessPlugin],
        base_dir: Optional[str] = None,
    ) -> PreprocessPlugin:
        if isinstance(spec, PreprocessPlugin):
            return spec

        if isinstance(spec, str):
            plugin_cls = cls.get_preprocess_class(spec)
            if plugin_cls:
                return plugin_cls()
            # Try loading as dynamic source
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PreprocessPlugin, base_dir=base_dir
            )
            return loaded_cls()

        if isinstance(spec, dict):
            plugin_key = spec.get("plugin") or spec.get("name")
            params = spec.get("params", {})
            if plugin_key:
                plugin_cls = cls.get_preprocess_class(plugin_key)
                if plugin_cls:
                    return plugin_cls(**params)
            if spec.get("name"):
                plugin_cls = cls.get_preprocess_class(spec["name"])
                if plugin_cls:
                    return plugin_cls(**params)
            # Try loading from script or module
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PreprocessPlugin, base_dir=base_dir
            )
            return loaded_cls(**params)

        raise PipelineError(f"Cannot resolve preprocess plugin from: {spec}")

    @classmethod
    def resolve_postprocess_plugin(
        cls,
        spec: Union[str, Dict[str, Any], PostprocessPlugin],
        base_dir: Optional[str] = None,
    ) -> PostprocessPlugin:
        if isinstance(spec, PostprocessPlugin):
            return spec

        if isinstance(spec, str):
            plugin_cls = cls.get_postprocess_class(spec)
            if plugin_cls:
                return plugin_cls()
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PostprocessPlugin, base_dir=base_dir
            )
            return loaded_cls()

        if isinstance(spec, dict):
            plugin_key = spec.get("plugin") or spec.get("name")
            params = spec.get("params", {})
            if plugin_key:
                plugin_cls = cls.get_postprocess_class(plugin_key)
                if plugin_cls:
                    return plugin_cls(**params)
            if spec.get("name"):
                plugin_cls = cls.get_postprocess_class(spec["name"])
                if plugin_cls:
                    return plugin_cls(**params)
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PostprocessPlugin, base_dir=base_dir
            )
            return loaded_cls(**params)

        raise PipelineError(f"Cannot resolve postprocess plugin from: {spec}")


# Decorator helpers for external developer contributions
def register_preprocess_plugin(
    name: str,
) -> Callable[[Type[PreprocessPlugin]], Type[PreprocessPlugin]]:
    def decorator(cls: Type[PreprocessPlugin]) -> Type[PreprocessPlugin]:
        PluginRegistry.register_preprocess(name, cls)
        return cls

    return decorator


def register_postprocess_plugin(
    name: str,
) -> Callable[[Type[PostprocessPlugin]], Type[PostprocessPlugin]]:
    def decorator(cls: Type[PostprocessPlugin]) -> Type[PostprocessPlugin]:
        PluginRegistry.register_postprocess(name, cls)
        return cls

    return decorator


def register_simulation_engine(
    name: str,
) -> Callable[[Type[BaseSimulationEngine]], Type[BaseSimulationEngine]]:
    def decorator(cls: Type[BaseSimulationEngine]) -> Type[BaseSimulationEngine]:
        PluginRegistry.register_engine(name, cls)
        return cls

    return decorator


register_engine = register_simulation_engine
