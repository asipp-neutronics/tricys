import os
import sys

# Force UTF-8 mode on Windows to prevent GBK decoding errors in OMPython/OMC subprocesses
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

__version__ = "2.0.0"


# Backward compatibility alias for legacy handlers moved to v1_examples_legacy
def _register_legacy_handlers_alias():
    try:
        from tricys.example.example_data.v1_examples_legacy import (
            handlers as legacy_handlers,
        )

        sys.modules.setdefault("tricys.handlers", legacy_handlers)
        import tricys as _pkg

        setattr(_pkg, "handlers", legacy_handlers)
        if hasattr(legacy_handlers, "i_iss_handler"):
            sys.modules.setdefault(
                "tricys.handlers.i_iss_handler", legacy_handlers.i_iss_handler
            )
        if hasattr(legacy_handlers, "div_handler"):
            sys.modules.setdefault(
                "tricys.handlers.div_handler", legacy_handlers.div_handler
            )
    except Exception:
        pass


_register_legacy_handlers_alias()
