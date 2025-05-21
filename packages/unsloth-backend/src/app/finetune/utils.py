import ast
import logging
import logging.config
import traceback

from omegaconf import OmegaConf


def setup_logger(name: str) -> logging.Logger:
    """Sets up and returns a named logger configured via YAML or falls back to basic logging.

    Attempts to read `conf/logging.yaml` for a structured configuration under `logging:`.
    If that fails, initializes a basic logger at INFO level and logs the error.

    Args:
        name (str): The namespace/name for the logger.

    Returns:
        logging.Logger: A logger instance configured per the YAML file or basic settings on error.
    """
    try:
        cfg = OmegaConf.load("conf/logging.yaml")
        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        logging.config.dictConfig(logging_config)
        return logging.getLogger(name)
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(name)
        logger.error(f"Failed to load logging configuration: {e}. Using basic logging.")
        logger.error(traceback.format_exc())
        return logger


def parse_metadata(headers: dict) -> dict:
    """Parses S3 object metadata headers, converting strings to bool, int, float, list, or leaving as str.

    Iterates over keys starting with `x-amz-meta-` and attempts:
      - Boolean casting for "true"/"false"
      - Integer casting for digit strings (with optional leading minus)
      - Float casting for numeric strings with at most one decimal point
      - List parsing via `ast.literal_eval` for bracket-delimited strings
      - Fallback to the original string if no casting applies or parsing fails

    Args:
        headers (dict): The HTTP headers dict from an S3 `head_object` response.

    Returns:
        dict: A mapping of metadata keys to values typed as bool, int, float, list, or str.
    """
    cfg = {}
    for key, value in headers.items():
        if not key.startswith("x-amz-meta-"):
            continue

        if value.lower() in ("true", "false"):
            cfg[key] = (value.lower() == "true")
        elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            cfg[key] = int(value)
        elif value.replace(".", "", 1).isdigit() or (
            value.startswith("-") and value[1:].replace(".", "", 1).isdigit()
        ):
            cfg[key] = float(value)
        elif value.startswith("[") and value.endswith("]"):
            try:
                cfg[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                cfg[key] = value
        else:
            cfg[key] = value

    return cfg
