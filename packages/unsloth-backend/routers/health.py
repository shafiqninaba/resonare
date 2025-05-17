import torch
from fastapi import APIRouter, HTTPException
from typing import Dict

from app.dependencies import resources

router = APIRouter(tags=["system"])


@router.get("/health")
async def get_health_status() -> Dict[str, str]:
    """Health check endpoint that verifies service readiness.

    Returns:
        Dict[str, str]: Service health status including S3 and GPU availability.
    """
    try:
        s3_status = "connected" if "s3_client" in resources else "disconnected"
        gpu_status = "available" if torch.cuda.is_available() else "unavailable"
        gpu_info = (
            f"{torch.cuda.device_count()} device(s)" if torch.cuda.is_available() else "none"
        )

        if s3_status == "disconnected":
            raise HTTPException(status_code=503, detail="S3 client not connected")

        return {
            "status": "healthy",
            "message": "API is operational",
            "s3_connection": s3_status,
            "gpu_status": gpu_status,
            "gpu_info": gpu_info,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))