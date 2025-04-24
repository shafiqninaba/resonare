import torch
import logging
from omegaconf import OmegaConf, DictConfig
import hydra
import traceback


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize logging
    logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("check_gpu")

    # Check if GPU is available
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")

    logger.info("Checking unsloth import")
    try:
        logger.info("unsloth imported successfully")
    except Exception as e:
        logger.error("Error importing unsloth")
        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    main()
