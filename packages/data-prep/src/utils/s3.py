import logging
import os

import boto3
from botocore.exceptions import ClientError
from typing import Optional

logger = logging.getLogger(__name__)


def setup_s3_client(
    region_name: str = "ap-southeast-1",
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
) -> boto3.client:
    """
    Set up an S3 client for uploading processed data to AWS S3.

    Args:
        region_name (str): AWS region name (e.g., "us-east-1").
        access_key (Optional[str]): AWS access key ID. Defaults to environment variable AWS_ACCESS_KEY_ID.
        secret_key (Optional[str]): AWS secret access key. Defaults to environment variable AWS_SECRET_ACCESS_KEY.

    Returns:
        boto3.client: Boto3 S3 client object.
    """
    try:
        # Use provided credentials or fall back to environment variables
        access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")

        if not region_name:
            raise ValueError("region_name is required to set up the S3 client.")

        # Create S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )
        logger.info(f"S3 client setup for {region_name} successful.")
        return s3

    except Exception as e:
        logger.error(
            f"Failed to set up S3 client: {e}, S3 upload will be skipped, only local export will be used."
        )
        raise


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
