from src.general_utils import upload_directory_to_s3, setup_s3_client, setup_logger
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    # Set up logger
    logger = setup_logger("test_upload_to_s3")
    # Example usage
    local_directory = "logs"
    s3_prefix = "b28c5ef19549486c9f09544ea1428162/logs"

    s3 = setup_s3_client()

    upload_directory_to_s3(local_directory, s3_prefix, s3, os.getenv("AWS_S3_BUCKET"))
