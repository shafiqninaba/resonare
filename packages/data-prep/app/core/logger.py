# app/core/logger.py
import logging

from app.utils.general import setup_standard_logging


def setup_logging():
    """
    Initialize application‐wide logging using conf/logging.yaml.
    Call this once at startup (e.g. in main.py).
    """
    setup_standard_logging(logging_config_path="conf/logging.yaml")

    # Propagate handlers into uvicorn’s logger
    uvicorn_log = logging.getLogger("uvicorn")
    uvicorn_log.handlers = logging.getLogger().handlers
