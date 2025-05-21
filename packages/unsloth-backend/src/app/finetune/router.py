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
    """Enqueue a new fine-tuning job.

    Args:
        req (FineTuneRequest): The request body containing the run_id.
        job_service (JobService): Dependency-injected job service.

    Returns:
        FineTuneResponse: Confirmation that the job has been queued.

    Raises:
        HTTPException: If the job cannot be enqueued (e.g., duplicate run_id).
    """
    return await job_service.submit(req)


@router.get(
    "/jobs",
    response_model=Dict[str, JobInfo],
    summary="List all fine-tuning jobs",
)
async def list_jobs(
    job_service: JobService = Depends(get_job_service),
) -> Dict[str, JobInfo]:
    """Retrieve a mapping of all fine-tuning jobs and their statuses.

    Args:
        job_service (JobService): Dependency-injected job service.

    Returns:
        Dict[str, JobInfo]: A dict mapping run_id to its JobInfo.
    """
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
    """Fetch the status and metadata for a specific fine-tuning job.

    Args:
        run_id (str): The unique identifier of the job.
        job_service (JobService): Dependency-injected job service.

    Returns:
        JobInfo: The metadata and status of the specified job.

    Raises:
        HTTPException: If the job is not found (404).
    """
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
    """Get the current queue status and running flag.

    Args:
        job_service (JobService): Dependency-injected job service.

    Returns:
        Dict[str, Any]: A dict containing:
            - running (bool): Whether a job is currently running.
            - queue_size (int): Number of jobs waiting in the queue.
            - jobs (Dict[str, JobInfo]): All job statuses.
    """
    return {
        "running": job_service.running,
        "queue_size": job_service.queue.qsize(),
        "jobs": job_service.statuses,
    }
