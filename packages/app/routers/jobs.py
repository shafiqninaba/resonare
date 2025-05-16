from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.dependencies import job_queue, job_status, job_running, resources, logger
from app.models import JobInfo, JobStatus, RunIDRequest, TrainingResponse

router = APIRouter(tags=["jobs"])


@router.post("/fine-tune", response_model=TrainingResponse)
async def fine_tune(request: RunIDRequest):
    """
    Start a fine-tuning job with the specified run ID

    The job will be queued and processed when resources are available
    """
    run_id = request.run_id

    # Validate run_id
    if not run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id provided")

    # Validate S3 client exists
    if "s3_client" not in resources:
        raise HTTPException(status_code=500, detail="S3 client not initialized")

    # Check if job already exists
    if run_id in job_status:
        raise HTTPException(
            status_code=409,
            detail=f"Job with run_id {run_id} already exists with status {job_status[run_id].status}",
        )

    # Create job info and add to queue
    job_info = JobInfo(
        run_id=run_id,
        status=JobStatus.QUEUED,
        position_in_queue=job_queue.qsize() + 1 if job_running else 0,
        created_at=datetime.now(),
    )
    job_status[run_id] = job_info
    await job_queue.put(run_id)

    return TrainingResponse(
        status="queued",
        message=f"Fine-tuning job queued at position {job_info.position_in_queue}",
        run_id=run_id,
    )


@router.get("/jobs")
async def get_all_job_status() -> Dict[str, JobInfo]:
    """Return the full mapping of all job statuses.

    Returns:
        Dict[str, JobInfo]: All known jobs, regardless of status.
    """
    return job_status


@router.get("/jobs/{run_id}")
async def get_job_status(run_id: str) -> JobInfo:
    """Retrieve the status and metadata of a specific preprocessing job.

    Args:
        run_id (str): Unique job identifier.

    Returns:
        JobInfo: Metadata about the specified job.

    Raises:
        HTTPException: If job ID is not found.
    """
    if run_id not in job_status:
        raise HTTPException(
            status_code=404, detail=f"Job with run_id {run_id} not found"
        )
    return job_status[run_id]


@router.get("/queue")
async def get_queue_status() -> Dict[str, Any]:
    """Get the status of all jobs currently in the queue.

    Returns:
        Dict[str, Any]: Job queue summary including queue size and all job statuses.
    """
    return {
        "running": job_running,
        "queue_size": job_queue.qsize(),
        "jobs": job_status,
    }