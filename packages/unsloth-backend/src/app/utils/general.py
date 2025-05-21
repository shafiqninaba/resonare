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
        cfg = OmegaConf.load("/app/conf/logging.yaml")
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
