import logging
import logging.config
import os

import boto3
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()


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
        return logger


logger = setup_logger(__name__)


def setup_s3_client() -> boto3.client:
    """
    Set up an S3 client for uploading processed data to AWS S3.

    Returns:
        s3: Boto3 S3 client object.
    """

    logger.info("Setting up S3 client for uploading processed data...")

    try:
        # Read AWS credentials from environment variables
        # Note: boto3 looks for AWS credentials in serveal locations. https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

        AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        AWS_REGION = os.getenv("AWS_REGION")
        AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

        if (
            not AWS_ACCESS_KEY
            or not AWS_SECRET_KEY
            or not AWS_REGION
            or not AWS_S3_BUCKET
        ):
            logger.error(
                "Missing AWS credentials in environment variables. "
                "Please set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and AWS_S3_BUCKET."
            )
            raise EnvironmentError("Missing AWS credentials in environment variables.")

        # Create S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
        logger.info("S3 client setup successful.")
        return s3

    except Exception as e:
        logger.error(
            f"Failed to set up S3 client: {e}, s3 upload will be skipped, only local export will be used."
        )


def downloadDirectoryFroms3(bucketName, remoteDirectoryName, localDirectoryName):
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket(bucketName)
    logger.info(
        f"Downloading {remoteDirectoryName} from S3 bucket {bucketName} to local directory {localDirectoryName}"
    )

    # Make sure the local directory exists
    os.makedirs(localDirectoryName, exist_ok=True)

    for obj in bucket.objects.filter(Prefix=remoteDirectoryName):
        # Skip directory markers
        if obj.key.endswith("/"):
            continue

        # Extract the filename from the object key
        file_name = obj.key.split("/")[-1]

        # Build the local file path
        local_file_path = os.path.join(localDirectoryName, file_name)

        # Download the file
        logger.info(f"Downloading {obj.key} to {local_file_path}")
        bucket.download_file(obj.key, local_file_path)
