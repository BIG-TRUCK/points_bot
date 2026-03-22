import json
import logging
import os
from datetime import datetime
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Custom logging formatter that outputs log records as JSON."""

    def format(self, record) -> str:
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_entry)


def setup_logging() -> None:
    """Sets up logging configuration for the application."""
    log_dir: str = os.path.join(os.path.expanduser('~'), 'points_bot')
    os.makedirs(log_dir, exist_ok=True)
    log_file: str = os.path.join(log_dir, f'logs_{datetime.now().strftime("%Y%m%d")}.jsonl')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Avoid adding duplicate handlers
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == handler.baseFilename for h in root_logger.handlers):
        root_logger.addHandler(handler)


# Call setup_logging when this module is imported
setup_logging()