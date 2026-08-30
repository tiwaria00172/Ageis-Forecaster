# =============================================================================
# Logger — Structured logging for the forecasting pipeline
# =============================================================================
"""
Provides a consistent logging interface across all modules.
Logs to both console and file with configurable verbosity.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a named logger with console and file handlers.

    Args:
        name: Logger name (typically module __name__).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        console: Whether to log to stdout.
        file: Whether to log to a file.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(log_dir, f"forecasting_{timestamp}.log")
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
