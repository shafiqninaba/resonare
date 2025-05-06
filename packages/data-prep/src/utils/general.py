#!/usr/bin/env python3
"""
General utility functions.

This module provides helper functions for:
- Setting up logging configurations (standard or Loguru).
"""

import logging
import os
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def setup_standard_logging(
    logging_config_path="conf/logging.yaml",
    default_level=logging.INFO,
):
    """Set up configuration for logging utilities."""

    # Get absolute path to project root from the logging config file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Ensure logs directory exists
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    try:
        with open(logging_config_path, encoding="utf-8") as file:
            log_config = yaml.safe_load(file)

        # Inject absolute paths into handlers (overwrite relative)
        for handler_name in [
            "debug_file_handler",
            "info_file_handler",
            "error_file_handler",
        ]:
            if handler_name in log_config.get("handlers", {}):
                filename = log_config["handlers"][handler_name]["filename"]
                log_config["handlers"][handler_name]["filename"] = os.path.join(
                    logs_dir, filename
                )

        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logging.getLogger().warning(error)
        logging.getLogger().warning(
            "Logging config file not found or invalid. Basic config is being used."
        )


def setup_loguru_logging(logs_dir="logs"):
    """Set up Loguru logging configuration for console and rotating file handlers.

    Args:
        logs_dir (str): Directory to save log files. Defaults to "logs".

    Raises:
        Exception: Any unexpected error during setup will trigger basic logging fallback.
    """

    try:
        from loguru import logger

        # Ensure logs directory exists
        logs_path = Path(logs_dir)
        logs_path.mkdir(parents=True, exist_ok=True)

        # Remove any existing Loguru handlers
        logger.remove()

        # --- 1. Pretty Console Handler ---
        logger.add(
            sink=sys.stdout,
            level="INFO",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
            backtrace=True,  # Show full stack trace
            diagnose=True,  # Display variable values in tracebacks
        )

        # --- 2. Debug File Handler ---
        logger.add(
            sink=str(logs_path / "debug.log"),
            level="DEBUG",
            rotation="10 MB",  # Rotate after 10MB
            retention="30 days",  # Keep for 30 days
            compression="zip",  # Compress old logs
            encoding="utf-8",
            serialize=False,  # Set True if JSON logs are needed
        )

        # --- 3. Info File Handler ---
        logger.add(
            sink=str(logs_path / "info.log"),
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            serialize=False,
        )

        # --- 4. Error File Handler ---
        logger.add(
            sink=str(logs_path / "error.log"),
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            encoding="utf-8",
            serialize=False,
        )

        logger.success("Loguru logging is fully set up.")

    except Exception as error:
        # --- Basic fallback in case Loguru setup fails ---
        import logging

        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        fallback_logger = logging.getLogger(__name__)
        fallback_logger.warning(error)
        fallback_logger.warning(
            "Failed to fully configure Loguru logging. Fallback basic logging is now active."
        )
