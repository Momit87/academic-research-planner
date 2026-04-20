"""
core/logging.py
===============
Structured logging configuration for the Academic Research Planner.

Design:
- Development: human-readable colored output
- Production: JSON-structured output (parseable by Datadog, CloudWatch, etc.)
- Every log line carries thread_id, phase, node, latency_ms where available
- LangSmith run IDs attached when tracing is active

Usage:
    from core.logging import get_logger
    logger = get_logger(__name__)

    # Basic log
    logger.info("Node started", extra={"thread_id": thread_id, "node": "main_agent"})

    # With latency
    logger.info(
        "Qdrant query completed",
        extra={"thread_id": thread_id, "latency_ms": 142, "chunks_returned": 8}
    )
"""

import json
import logging
import sys
import time
from typing import Any

from core.config import get_settings


# ------------------------------------------------------------------
# JSON Formatter — used in production
# ------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    All `extra` fields passed to the logger are included at the top level.

    Example output:
        {
            "timestamp": "2026-04-20T11:00:00.000Z",
            "level": "INFO",
            "logger": "llm.workflow.research_planner.node.main_agent",
            "message": "Node entry",
            "thread_id": "abc-123",
            "phase": "clustering",
            "latency_ms": 342
        }
    """

    # Fields that are standard on every LogRecord — exclude from extra
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname",
        "filename", "module", "exc_info", "exc_text", "stack_info",
        "lineno", "funcName", "created", "msecs", "relativeCreated",
        "thread", "threadName", "processName", "process", "message",
        "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Attach any extra fields passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value

        # Attach exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ------------------------------------------------------------------
# Human-readable Formatter — used in development
# ------------------------------------------------------------------

class DevFormatter(logging.Formatter):
    """
    Colored, human-readable formatter for development terminals.

    Format:
        [HH:MM:SS] LEVEL  logger.name  — message  {extra fields}
    """

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    _RESERVED = JSONFormatter._RESERVED

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        color = self.COLORS.get(record.levelname, "")

        # Collect extra fields
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in self._RESERVED
        }
        extras_str = f"  {extras}" if extras else ""

        line = (
            f"{color}[{self.formatTime(record, '%H:%M:%S')}] "
            f"{record.levelname:<8}{self.RESET} "
            f"\033[90m{record.name}\033[0m — "
            f"{record.message}"
            f"\033[90m{extras_str}\033[0m"
        )

        if record.exc_info:
            line += f"\n{self.formatException(record.exc_info)}"

        return line


# ------------------------------------------------------------------
# Setup function — called once at application startup
# ------------------------------------------------------------------

def setup_logging() -> None:
    """
    Configure the root logger and all relevant library loggers.

    Call this once in main.py before anything else runs.
    Subsequent calls are no-ops (idempotent via handler check).
    """
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()

    # Idempotent — don't add duplicate handlers on reload
    if root_logger.handlers:
        return

    root_logger.setLevel(level)

    # Choose formatter based on environment
    if settings.app_env == "production":
        formatter = JSONFormatter()
    else:
        formatter = DevFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Silence noisy third-party loggers
    # ------------------------------------------------------------------
    _set_level("httpx", logging.WARNING)
    _set_level("httpcore", logging.WARNING)
    _set_level("urllib3", logging.WARNING)
    _set_level("asyncio", logging.WARNING)
    _set_level("qdrant_client", logging.WARNING)
    _set_level("langchain", logging.WARNING)       # very verbose at DEBUG
    _set_level("openai", logging.WARNING)
    _set_level("anthropic", logging.WARNING)
    _set_level("cohere", logging.WARNING)

    # LangSmith: keep at INFO so trace URLs appear in logs
    _set_level("langsmith", logging.INFO)


def _set_level(logger_name: str, level: int) -> None:
    """Set level on a named logger without adding handlers."""
    logging.getLogger(logger_name).setLevel(level)


# ------------------------------------------------------------------
# Factory — used everywhere in the project
# ------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("Starting ingestion", extra={"thread_id": thread_id})

    Args:
        name: typically __name__ from the calling module

    Returns:
        logging.Logger: configured logger instance
    """
    return logging.getLogger(name)


# ------------------------------------------------------------------
# Timing context manager — for latency logging
# ------------------------------------------------------------------

class timer:
    """
    Context manager that measures elapsed time and logs it.

    Usage:
        with timer("qdrant_upsert", logger, extra={"thread_id": tid}):
            await qdrant.upsert(...)

    Logs on exit:
        INFO qdrant_upsert completed  {"latency_ms": 142, "thread_id": "abc"}
    """

    def __init__(
        self,
        operation: str,
        logger: logging.Logger,
        level: int = logging.INFO,
        extra: dict[str, Any] | None = None,
    ):
        self.operation = operation
        self.logger = logger
        self.level = level
        self.extra = extra or {}
        self._start: float = 0.0

    def __enter__(self) -> "timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        extra = {**self.extra, "latency_ms": elapsed_ms}

        if exc_type is not None:
            self.logger.error(
                f"{self.operation} failed",
                extra={**extra, "error": str(exc_val)},
                exc_info=True,
            )
        else:
            self.logger.log(self.level, f"{self.operation} completed", extra=extra)