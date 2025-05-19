# app/core/s3_client.py
import os
from functools import lru_cache
from logging import getLogger

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

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
    if not all([settings.AWS_ACCESS_KEY_ID, settings.AWS_SECRET_ACCESS_KEY, settings.AWS_REGION]):
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
