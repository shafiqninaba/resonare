# app/system/router.py
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.s3_client import get_s3_client

router = APIRouter(tags=["system"], prefix="/system")


@router.get("/health")
async def get_health_status() -> dict:
    """Perform a health check on the API and its S3 connection.

    This endpoint always returns API status as "up" and attempts a
    lightweight S3 `HeadBucket` call to verify connectivity.

    Returns:
        dict: A JSON object containing:
            - api (str): Always "up".
            - s3_bucket (str): The name of the S3 bucket being checked.
            - s3_status (str): "connected" if the bucket is reachable.

    Raises:
        HTTPException (503): If the S3 `HeadBucket` call fails, returns a
            503 Service Unavailable with the error detail.
    """
    s3 = get_s3_client()
    try:
        # Lightweight call to confirm bucket exists & is accessible
        s3.head_bucket(Bucket=settings.AWS_S3_BUCKET)
        s3_status = "connected"
    except (ClientError, BotoCoreError) as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"S3 health check failed: {e}",
        )

    return {
        "api": "up",
        "s3_bucket": settings.AWS_S3_BUCKET,
        "s3_status": s3_status,
    }

