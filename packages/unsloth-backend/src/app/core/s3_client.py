# app/core/s3_client.py
import os
from functools import lru_cache
from logging import getLogger

import boto3
from botocore.exceptions import ClientError

from .config import settings

logger = getLogger(__name__)


@lru_cache()
def get_s3_client() -> boto3.client:
    """
    Returns a cached boto3 S3 client configured with credentials from Settings.
    If you later switch to aioboto3/aiobotocore for true async, just replace this function.

    Returns:
        boto3.client: Boto3 S3 client object.
    """

    # Check if the required environment variables are set
    if not all(
        [
            settings.AWS_ACCESS_KEY_ID,
            settings.AWS_SECRET_ACCESS_KEY,
            settings.AWS_REGION,
        ]
    ):
        raise ValueError(
            "AWS credentials and region must be set in the environment variables."
        )

    try:
        # Check if the S3 bucket exists
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        s3_client.head_bucket(Bucket=settings.AWS_S3_BUCKET)
    except ClientError as e:
        raise RuntimeError(f"Failed to connect to S3: {e}")

    return s3_client


# Helper function to upload directory content to S3
def upload_directory_to_s3(directory_path, s3_prefix, s3_client, AWS_S3_BUCKET):
    """
    Uploads the contents of a local directory to an S3 bucket.

    Args:
        directory_path (str): Path to the local directory to upload.
        s3_prefix (str): S3 prefix (path) where the files will be uploaded.
        s3_client (boto3.client): Boto3 S3 client object.
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
                s3_client.upload_file(local_file_path, AWS_S3_BUCKET, s3_object_name)
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


def list_directories_in_bucket(s3_client, bucket_name: str) -> list[str]:
    """
    Lists all top-level directories in a given S3 bucket.

    Args:
        s3_client (boto3.client): Boto3 S3 client object.
        bucket_name (str): The name of the S3 bucket.

    Returns:
        list[str]: A list of directory names (prefixes ending with '/').
    """
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
