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


def list_directories_in_bucket(bucket_name: str) -> list[str]:
    """
    Lists all top-level directories in a given S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.

    Returns:
        list[str]: A list of directory names (prefixes ending with '/').
    """
    s3_client = boto3.client("s3")
    directories = (
        set()
    )  # Use a set to avoid duplicates if pagination returns overlapping prefixes
    paginator = s3_client.get_paginator("list_objects_v2")
    logger.info(f"Listing directories in S3 bucket: {bucket_name}")

    try:
        # Paginate through results to handle buckets with many objects/prefixes
        page_iterator = paginator.paginate(Bucket=bucket_name, Delimiter="/")
        for page in page_iterator:
            if "CommonPrefixes" in page:
                for prefix_info in page["CommonPrefixes"]:
                    # CommonPrefixes gives paths ending with the delimiter, e.g., 'folder/'
                    directories.add(
                        prefix_info["Prefix"].rstrip("/")
                    )  # Remove trailing slash for consistency
                    logger.debug(f"Found directory: {prefix_info['Prefix']}")

        logger.info(f"Found {len(directories)} directories in bucket {bucket_name}.")
        return sorted(list(directories))  # Return sorted list for consistency

    except Exception as e:
        logger.error(f"Failed to list directories in bucket {bucket_name}: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    bucket_name = os.getenv("AWS_S3_BUCKET")
    if bucket_name:
        directories = list_directories_in_bucket(bucket_name)
        print("Directories in bucket:", directories)
    else:
        print("Bucket name not set in environment variables.")
