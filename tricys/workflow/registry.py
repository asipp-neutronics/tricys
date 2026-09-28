from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

from tricys.workflow.base import (
    BaseSimulationEngine,
    PipelineError,
    PostprocessPlugin,
    PreprocessPlugin,
)

logger = logging.getLogger(__name__)


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
    def load_class_from_source(
        cls,
        source: Union[str, Dict[str, Any]],
        expected_base: Optional[Type[Any]] = None,
    ) -> Type[Any]:
        """Dynamically load a class from a script path or python module path.

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
            script_path_str = source.get("script_path")
            module_name_str = source.get("module")
            class_name = source.get("class") or source.get("class_name")
        elif isinstance(source, str):
            if source.endswith(".py") or "\\" in source or "/" in source:
                script_path_str = source
            else:
                module_name_str = source

        loaded_module = None
        if script_path_str:
            path = Path(script_path_str).resolve()
            if not path.is_file():
                raise PipelineError(f"Plugin script not found at {path}")
            mod_name = path.stem
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if not spec or not spec.loader:
                raise PipelineError(f"Failed to create module spec for {path}")
            loaded_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded_module)
        elif module_name_str:
            if "." in module_name_str and not class_name:
                parts = module_name_str.split(".")
                # Check if last part is capitalized (likely class name)
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
        cls, spec: Union[str, Dict[str, Any], PreprocessPlugin]
    ) -> PreprocessPlugin:
        if isinstance(spec, PreprocessPlugin):
            return spec

        if isinstance(spec, str):
            plugin_cls = cls.get_preprocess_class(spec)
            if plugin_cls:
                return plugin_cls()
            # Try loading as dynamic source
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PreprocessPlugin
            )
            return loaded_cls()

        if isinstance(spec, dict):
            name = spec.get("name")
            params = spec.get("params", {})
            if name:
                plugin_cls = cls.get_preprocess_class(name)
                if plugin_cls:
                    return plugin_cls(**params)
            # Try loading from script_path or module
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PreprocessPlugin
            )
            return loaded_cls(**params)

        raise PipelineError(f"Cannot resolve preprocess plugin from: {spec}")

    @classmethod
    def resolve_postprocess_plugin(
        cls, spec: Union[str, Dict[str, Any], PostprocessPlugin]
    ) -> PostprocessPlugin:
        if isinstance(spec, PostprocessPlugin):
            return spec

        if isinstance(spec, str):
            plugin_cls = cls.get_postprocess_class(spec)
            if plugin_cls:
                return plugin_cls()
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PostprocessPlugin
            )
            return loaded_cls()

        if isinstance(spec, dict):
            name = spec.get("name")
            params = spec.get("params", {})
            if name:
                plugin_cls = cls.get_postprocess_class(name)
                if plugin_cls:
                    return plugin_cls(**params)
            loaded_cls = cls.load_class_from_source(
                spec, expected_base=PostprocessPlugin
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
