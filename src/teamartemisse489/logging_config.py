"""Centralized logging configuration."""

from __future__ import annotations

import logging
import sys
from logging.config import dictConfig
from pathlib import Path
from typing import Literal

from rich.logging import RichHandler

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Advanced logging configuration with Rich JSON format
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},  # Rich adds its own formatting
        "detailed": {
            "format": (
                "%(levelname)s %(asctime)s "
                "[%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n"
                "%(message)s\n"
            )
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.INFO,
        },
        "debug": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "debug.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 8,
            "formatter": "detailed",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 8,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "warning": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "warning.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 8,
            "formatter": "detailed",
            "level": logging.WARN,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 8,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
        "critical": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "critical.log"),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 8,
            "formatter": "detailed",
            "level": logging.CRITICAL,
        },
    },
    "root": {
        "handlers": ["console", "debug", "info", "warning", "error", "critical"],
        "level": logging.DEBUG,
        "propagate": True,
    },
}

def setup_logging() -> None:
    """Configure the root logger for the application."""
    dictConfig(logging_config) #apply config
    logger = logging.getLogger(__name__) #get root logger
    #set handler to rich
    logger.root.handlers[0] = RichHandler(markup=True, rich_tracebacks=True)
    logger.info("[bold green]Info:[/bold green] \
Rich logging has activated Successfully!")


def test_logging() -> None :
    logger = logging.getLogger(__name__) #get root logger
    logger.debug("[bold]Debug:[/bold] Example Debug Message: \
Loading dataset with 10,000 samples")
    logger.info("[bold green]Info:[/bold green] Example Info Message: \
Model training started with learning rate=0.001")
    logger.warning("[bold yellow]Warning:[/bold yellow] Example Warning Message: \
Validation loss increasing, possible overfitting")
    logger.error("[bold red]Error:[/bold red] Example Error Message: \
Failed to save model weights to S3")
    logger.critical("[bold white on red]CRITICAL:[/bold white on red] \
Example Critical Message: Out of memory during batch processing")


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)

# example Here
setup_logging()
test_logging()
