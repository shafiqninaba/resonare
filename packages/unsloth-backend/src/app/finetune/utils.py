import ast
import logging
import logging.config
import traceback

from omegaconf import OmegaConf


def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger for the application.

    Args:
        name (str): Name of the logger.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    try:
        # Load logging configuration from YAML file
        cfg = OmegaConf.load("conf/logging.yaml")
        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)

        # Initialize logging
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger(name)
        return logger
    except Exception as e:
        # Fallback to basic logging if YAML loading fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(name)
        logger.error(f"Failed to load logging configuration: {e}. Using basic logging.")
        logger.error(traceback.format_exc())
        return logger


def parse_metadata(headers):
    """Parse S3 metadata headers and convert to appropriate types"""
    cfg = {}
    for key, value in headers.items():
        if key.startswith("x-amz-meta-"):
            clean_key = key

            # Handle booleans
            if value.lower() in ("true", "false"):
                cfg[clean_key] = value.lower() == "true"

            # Handle integers
            elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                cfg[clean_key] = int(value)

            # Handle floats
            elif value.replace(".", "", 1).isdigit() or (
                value.startswith("-") and value[1:].replace(".", "", 1).isdigit()
            ):
                cfg[clean_key] = float(value)

            # Handle lists
            elif value.startswith("[") and value.endswith("]"):
                try:
                    cfg[clean_key] = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    # Fallback to string if parsing fails
                    cfg[clean_key] = value

            # Keep as string for other values
            else:
                cfg[clean_key] = value

    return cfg
