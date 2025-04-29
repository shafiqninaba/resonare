import logging
from omegaconf import OmegaConf
import logging.config


def setup_logger(name):
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
        return logger


logger = setup_logger(__name__)


def setup_s3_client():
    """
    Set up an S3 client for uploading processed data to AWS S3.

    Args:
        cfg: Configuration object containing AWS credentials and S3 bucket information.

    Returns:
        s3: Boto3 S3 client object.
    """

    logger.info("Setting up S3 client for uploading processed data...")

    # try:
    #     # Read AWS credentials from environment variables
    #     # Note: boto3 looks for AWS credentials in serveal locations. https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    #     AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    #     AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    #     if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    #         raise EnvironmentError(
    #             "Missing AWS credentials in environment variables."
    #         )

    #     # Create S3 client
    #     s3 = boto3.client(
    #         "s3",
    #         aws_access_key_id=AWS_ACCESS_KEY,
    #         aws_secret_access_key=AWS_SECRET_KEY,
    #         region_name=cfg.output.s3_region,
    #     )
    #     logger.info("S3 client setup successful.")

    # except Exception as e:
    #     logger.error(
    #         f"Failed to set up S3 client: {e}, s3 upload will be skipped, only local export will be used."
    #     )
