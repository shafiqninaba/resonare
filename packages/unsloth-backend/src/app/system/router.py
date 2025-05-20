# src/app/system/router.py

from typing import Dict

import torch
from fastapi import APIRouter, Depends, HTTPException

from ..core.config import settings
from ..dependencies import get_s3_client_dep

router = APIRouter(tags=["system"], prefix="/system")


@router.get(
    "/health",
    response_model=Dict[str, str],
    summary="Health check for S3 connectivity & GPU availability",
)
async def get_health_status(
    s3_client=Depends(get_s3_client_dep),
) -> Dict[str, str]:
    """
    Health check endpoint that verifies:
      - S3 client can reach the bucket
      - GPU is available (and how many devices)
    """
    # 1) S3 connectivity
    try:
        s3_client.head_bucket(Bucket=settings.AWS_S3_BUCKET)
        s3_status = "connected"
    except Exception:
        raise HTTPException(status_code=503, detail="Cannot connect to S3")

    # 2) GPU availability
    if torch.cuda.is_available():
        gpu_status = "available"
        gpu_info = f"{torch.cuda.device_count()} device(s)"
    else:
        gpu_status = "unavailable"
        gpu_info = "none"

    return {
        "status": "healthy",
        "s3_connection": s3_status,
        "gpu_status": gpu_status,
        "gpu_info": gpu_info,
    }
