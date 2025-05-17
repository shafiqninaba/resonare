# app/modules/jobs/router.py
from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.modules.jobs.schemas import (
    DataPrepRequest,
    DataPrepRequestResponse,
    JobInfo,
)
from app.modules.jobs.services import JobService, get_job_service

router = APIRouter(tags=["jobs"], prefix="/jobs")


@router.post(
    "/submit",
    response_model=DataPrepRequestResponse,
    summary="Submit a new data prep job",
)
async def submit_job(
    req: DataPrepRequest,
    service: JobService = Depends(get_job_service),
) -> DataPrepRequestResponse:
    return await service.create_job(req)


@router.get(
    "/",
    summary="List all jobs, or get one by run_id query",
)
async def get_jobs(
    run_id: str | None = Query(
        None,
        description="If provided, fetch only this job by ID",
        regex="^[0-9a-f]{32}$",
    ),
    service: JobService = Depends(get_job_service),
) -> Union[Dict[str, JobInfo], JobInfo]:
    if run_id:
        info = service.get_status(run_id)
        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {run_id} not found",
            )
        return info

    return service.statuses


@router.get(
    "/queue",
    response_model=Dict[str, Any],
    summary="Inspect queue and running flag",
)
async def get_queue_status(
    service: JobService = Depends(get_job_service),
) -> Dict[str, Any]:
    return {
        "running": service.job_running,
        "queue_size": service.queue.qsize(),
        "jobs": service.statuses,
    }
