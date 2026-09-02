import os
import sys

# Force UTF-8 mode on Windows to prevent GBK decoding errors in OMPython/OMC subprocesses
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
