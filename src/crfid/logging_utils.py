"""Structured logging helpers without import-time configuration."""

from __future__ import annotations

import json
import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = context
        return json.dumps(payload, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the package logger on explicit request."""

    logger = logging.getLogger("crfid")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger without causing output or file access."""

    return logging.getLogger(f"crfid.{name}")
