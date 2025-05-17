# app/modules/system/router.py
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.s3_client import get_s3_client

router = APIRouter(tags=["system"], prefix="/system")
settings = get_settings()


@router.get("/health")
async def get_health_status():
    """
    Returns:
      - API: always up
      - S3: tries a HeadBucket to verify connectivity
    """
    s3 = get_s3_client()
    try:
        # lightweight call to confirm bucket exists & is accessible
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
