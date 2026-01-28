import json
import logging
from typing import Any, Dict, List, Optional


class MemoryLogHandler(logging.Handler):
    """
    A logging handler that stores log records in memory.
    """

    def __init__(self, capacity: int = 10000):
        super().__init__()
        self.records: List[Dict[str, Any]] = []
        self.capacity = capacity
        # use a simple formatter or rely on the record attributes
        # We'll store structured data

    def emit(self, record: logging.LogRecord):
        if len(self.records) >= self.capacity:
            return

        try:
            self.format(record)
            log_entry = {
                "asctime": logging.Formatter().formatTime(record),
                "name": record.name,
                "levelname": record.levelname,
                "message": record.getMessage(),  # formatted message
                "lineno": record.lineno,
                "module": record.module,
            }
            # Add extra fields if available (from jsonlogger or extra={...})
            if hasattr(record, "props"):  # specialized checking
                pass

            self.records.append(log_entry)
        except Exception:
            self.handleError(record)

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.records

    def to_json(self) -> str:
        return json.dumps(self.records)


class LogCapture:
    """
    Context manager to capture logs.
    """

    def __init__(self, logger_name: Optional[str] = None):
        self.logger_name = logger_name
        self.handler = MemoryLogHandler()
        self.logger = logging.getLogger(logger_name)

    def __enter__(self):
        self.logger.addHandler(self.handler)
        return self.handler

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeHandler(self.handler)
