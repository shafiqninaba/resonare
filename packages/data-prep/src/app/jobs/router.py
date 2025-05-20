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
    return {
        "running": job_service.job_running,
        "queue_size": job_service.queue.qsize(),
        "jobs": job_service.statuses,
    }
