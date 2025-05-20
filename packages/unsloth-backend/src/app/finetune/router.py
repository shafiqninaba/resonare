from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Path, status

from .schemas import FineTuneRequest, FineTuneResponse, JobInfo
from .services import JobService, get_job_service

router = APIRouter(prefix="/fine-tune", tags=["fine-tuning"])


@router.post(
    "/",
    response_model=FineTuneResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a fine-tuning job",
)
async def submit_fine_tune(
    req: FineTuneRequest,
    job_service: JobService = Depends(get_job_service),
) -> FineTuneResponse:
    return await job_service.submit(req)


@router.get(
    "/jobs",
    response_model=Dict[str, JobInfo],
    summary="List all fine-tuning jobs",
)
async def list_jobs(
    job_service: JobService = Depends(get_job_service),
) -> Dict[str, JobInfo]:
    return job_service.statuses


@router.get(
    "/jobs/{run_id}",
    response_model=JobInfo,
    summary="Get status of one fine-tuning job",
)
async def get_job(
    run_id: str = Path(
        ..., pattern=r"^[0-9a-f]{32}$", description="The ID of the job to retrieve"
    ),
    job_service: JobService = Depends(get_job_service),
) -> JobInfo:
    job = job_service.get_status(run_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {run_id} not found",
        )
    return job


@router.get(
    "/queue",
    response_model=Dict[str, Any],
    summary="Inspect queue & running flag",
)
async def get_queue(
    job_service: JobService = Depends(get_job_service),
) -> Dict[str, Any]:
    return {
        "running": job_service.running,
        "queue_size": job_service.queue.qsize(),
        "jobs": job_service.statuses,
    }
