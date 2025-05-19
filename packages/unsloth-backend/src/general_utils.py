import logging
import logging.config
import os
import traceback

import boto3
from botocore.exceptions import ClientError
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
        logger.error(traceback.format_exc())
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


# Helper function to upload directory content to S3
def upload_directory_to_s3(directory_path, s3_prefix, s3, AWS_S3_BUCKET):
    """
    Uploads the contents of a local directory to an S3 bucket.

    Args:
        directory_path (str): Path to the local directory to upload.
        s3_prefix (str): S3 prefix (path) where the files will be uploaded.
        s3 (boto3.client): Boto3 S3 client object.
        AWS_S3_BUCKET (str): Name of the S3 bucket.

    Returns:
        bool: True if all files were uploaded successfully, False otherwise.
    """
    success = True
    for root, _, files in os.walk(directory_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Get the relative path to maintain directory structure
            relative_path = os.path.relpath(local_file_path, directory_path)
            s3_object_name = f"{s3_prefix}/{relative_path}"

            try:
                s3.upload_file(local_file_path, AWS_S3_BUCKET, s3_object_name)
                logger.info(
                    f"Uploaded {local_file_path} to s3://{AWS_S3_BUCKET}/{s3_object_name}"
                )
            except ClientError as e:
                logger.error(f"Failed to upload {local_file_path}: {e}")
                success = False
    return success


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
