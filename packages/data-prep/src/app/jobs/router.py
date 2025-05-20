# app/jobs/router.py
from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.jobs.services import JobService

from ..dependencies import get_job_service_dep
from .schemas import (
    DataPrepRequest,
    DataPrepRequestResponse,
    JobInfo,
)

router = APIRouter(tags=["jobs"], prefix="/jobs")


@router.post(
    "/submit",
    response_model=DataPrepRequestResponse,
    summary="Submit a new data prep job",
)
async def submit_job(
    req: DataPrepRequest,
    job_service: JobService = Depends(get_job_service_dep),  # Correct type hint and DI
) -> DataPrepRequestResponse:
    """
    Enqueue a new data preparation job for asynchronous processing.

    Args:
        req (DataPrepRequest): Pydantic model containing raw chats and optional preprocessing settings.
        job_service (JobService, dependency): Singleton service managing the job queue.

    Returns:
        DataPrepRequestResponse: Contains the generated run_id, status, and a human-readable message.

    Raises:
        HTTPException(409): If the generated run_id collides with an existing job.
        HTTPException(400): If the request payload is invalid (e.g. missing chats).
    """
    return await job_service.create_job(req)


@router.get(
    "/",
    summary="List all jobs, or get one by run_id query",
)
async def get_jobs(
    run_id: str | None = Query(
        None,
        description="If provided, fetch only this job by ID",
        pattern="^[0-9a-f]{32}$",
    ),
    job_service: JobService = Depends(get_job_service_dep),
) -> Union[Dict[str, JobInfo], JobInfo]:
    """
    Retrieve the status of data prep jobs.

    If `run_id` is provided, returns that single job’s info; otherwise returns
    the mapping of all run_id → JobInfo.

    Args:
        run_id (Optional[str]): 32-hex job identifier to look up.
        job_service (JobService, dependency): Singleton service managing the job queue.

    Returns:
        Union[Dict[str, JobInfo], JobInfo]:
            - When `run_id` is None → a dict of all known jobs.
            - When `run_id` is provided and exists → the corresponding JobInfo.

    Raises:
        HTTPException(404): If `run_id` is provided but no such job exists.
    """
    if run_id:
        info = job_service.get_status(run_id)
        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {run_id} not found",
            )
        return info

    return job_service.statuses


@router.get(
    "/queue",
    response_model=Dict[str, Any],
    summary="Inspect queue and running flag",
)
async def get_queue_status(
    job_service: JobService = Depends(get_job_service_dep),
) -> Dict[str, Any]:
    """
    Inspect the in-memory job queue and running state.

    Args:
        job_service (JobService, dependency): Singleton service managing the job queue.

    Returns:
        Dict[str, Any]: A summary object:
            - `running` (bool): True if a job is actively processing.
            - `queue_size` (int): Number of jobs waiting.
            - `jobs` (Dict[str, JobInfo]): Full run_id → JobInfo map.
    """
    return {
        "running": job_service.job_running,
        "queue_size": job_service.queue.qsize(),
        "jobs": job_service.statuses,
    }
