#!/usr/bin/env python3
"""
Chat‑Data‑Prep API
==================
POST /data-prep/process
GET  /data-prep/{run_id}/status
GET  /data-prep/queue
GET  /health
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from src.processing import run_data_processing
from src.utils.general import setup_standard_logging
from src.utils.s3 import setup_s3_client

# setup environmen variables and logging
load_dotenv()

project_root = Path(__file__).parent
logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")

setup_standard_logging(
    logging_config_path=os.path.join(
        project_root,
        "conf",
        "logging.yaml",
    ),
)


# models
class JobStatus(str, Enum):
    """Enum representing the current status of a job in the queue."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Metadata and lifecycle information for a single preprocessing job.

    Attributes:
        run_id (str): Unique identifier for the job.
        status (JobStatus): Current status of the job (e.g. queued, running).
        position_in_queue (Optional[int]): Position of the job in the queue, if applicable.
        created_at (datetime): Timestamp when the job was submitted.
        started_at (Optional[datetime]): Timestamp when processing started.
        completed_at (Optional[datetime]): Timestamp when processing completed or failed.
        error (Optional[str]): Error message if the job failed.
    """

    run_id: str
    status: JobStatus
    position_in_queue: Optional[int] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    """Standard response returned after submitting a preprocessing job.

    Attributes:
        status (str): Status of the request (e.g. "queued").
        message (str): Human-readable message for client feedback.
        run_id (str): Unique identifier for the submitted job.
    """

    status: str
    message: str
    run_id: str


# Dictionary to store connections and resources
resources: Dict = {}  # S3, input_specs, background‑task handle

# Create a job queue and job status tracking
job_queue: asyncio.Queue[str] = asyncio.Queue()
job_status: Dict[str, JobInfo] = {}
job_running = False


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup resources on startup
    logger.info("Setting up S3 client on application startup")
    resources["s3_client"] = setup_s3_client()
    resources["inputs"] = {}  # run_id → {"path": ...}
    logger.info("S3 client initialized successfully.")

    # Start the job processor
    resources["job_processor"] = asyncio.create_task(_worker())

    yield

    # Clean up resources on shutdown
    if "job_processor" in resources:
        resources["job_processor"].cancel()
        logger.info("Job processor cancelled")

    logger.info("Cleaning up resources")
    resources.clear()
    logger.info("Resources cleaned up")


# worker process
async def _worker():
    """Worker process that pulls jobs from the queue and runs them sequentially.

    This coroutine continuously:
      - Retrieves the next job from the `job_queue`.
      - Marks the job as running, updates the queue position, and starts processing.
      - Calls `run_data_processing` on a background thread.
      - On success, marks the job as completed and triggers an optional finetuning hook.
      - On failure, captures the error and updates job status accordingly.

    Returns:
        None
    """
    global job_running

    while True:
        try:
            run_id = await job_queue.get()
            job_info = job_status[run_id]
            job_info.status = JobStatus.RUNNING
            job_info.started_at = datetime.now()
            job_running = True

            # Update queue positions for remaining jobs
            update_queue_positions()

            logger.info(f"Starting data processing for job {run_id} from queue")

            spec = resources["inputs"].pop(run_id)

            # run the data processing in a separate thread
            try:
                # Run synchronously to block the queue processing until complete
                stats = await asyncio.to_thread(
                    run_data_processing, run_id, resources, spec
                )
                job_info.status = JobStatus.COMPLETED
                job_info.completed_at = datetime.now()
                job_info.stats = stats
                logger.info(f"Data processing for job {run_id} completed successfully")

                # # Post-processing trigger
                # logger.info(f"Triggering finetuning for job {run_id}")
                # await finetuning_hook(
                #     endpoint="http://fine-tuning:8001/fine-tune", run_id=run_id
                # )

            except Exception as e:
                job_info.status = JobStatus.FAILED
                job_info.error = str(e)
                logger.error(f"Job {run_id} failed: {str(e)}")

            finally:
                job_running = False
                job_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Job processor task was cancelled")
            break

        except Exception as e:
            logger.error(f"Error in job processor: {str(e)}")
            await asyncio.sleep(5)  # Wait before retrying


async def finetuning_hook(endpoint: str, run_id: str) -> None:
    """Triggers a downstream fine-tuning job via HTTP POST.

    Args:
        endpoint (str): URL of the fine-tuning service to notify.
        run_id (str): Unique job identifier for the completed preprocessing run.

    Returns:
        None

    Logs:
        - Status code of the downstream trigger.
        - Error details if the hook fails.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json={"run_id": run_id}, timeout=10)
            logger.info(
                f"[HOOK] Fine-tune trigger status for job {run_id}: {response.status_code}"
            )
    except Exception as e:
        logger.error(f"[HOOK] Failed to trigger fine-tune for job {run_id}: {e}")


def update_queue_positions() -> None:
    """Updates the `position_in_queue` field for all currently queued jobs.

    This ensures that clients polling for job status can view their
    relative position in the queue.

    Returns:
        None
    """
    queued_jobs = [
        job_id for job_id, info in job_status.items() if info.status == JobStatus.QUEUED
    ]

    for i, job_id in enumerate(queued_jobs):
        job_status[job_id].position_in_queue = i + 1


app = FastAPI(
    title="Data processing API",
    description="API for processing JSON data from Telegram chat exports.",
    lifespan=lifespan,
)


@app.post("/process", response_model=APIResponse)
async def submit_job(
    payload: Dict[str, Any] = Body(...),
) -> APIResponse:
    run_id = uuid.uuid4().hex

    if "s3_client" not in resources:
        raise HTTPException(status_code=500, detail="S3 client not initialized.")

    if run_id in job_status:
        raise HTTPException(
            status_code=409,
            detail=f"Job with run_id {run_id} already exists with status {job_status[run_id].status}",
        )

    chats = payload.get("chats")  # the actual merged chat data
    overrides = payload.get(
        "overrides", {}
    )  # any additional parameters for data processing

    if not isinstance(chats, (list, dict)):
        raise HTTPException(status_code=400, detail="chats must be a list or object")

    temp_dir = Path(tempfile.gettempdir()) / "chat_data_prep"
    temp_dir.mkdir(parents=True, exist_ok=True)

    raw_path = temp_dir / f"{run_id}.json"
    raw_path.write_text(json.dumps(chats), encoding="utf-8")

    resources["inputs"][run_id] = {
        "path": str(raw_path),
        "overrides": overrides,
    }

    job_status[run_id] = JobInfo(
        run_id=run_id,
        status=JobStatus.QUEUED,
        position_in_queue=job_queue.qsize() + (1 if job_running else 0),
        created_at=datetime.now(),
    )

    await job_queue.put(run_id)

    return APIResponse(
        status="queued",
        message="Job accepted and enqueued",
        run_id=run_id,
    )


@app.get("/jobs")
async def get_all_job_status() -> Dict[str, JobInfo]:
    """Return the full mapping of all job statuses.

    Returns:
        Dict[str, JobInfo]: All known jobs, regardless of status.
    """
    return job_status


@app.get("/jobs/{run_id}")
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


@app.get("/queue")
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


@app.get("/health")
async def get_health_status() -> Dict[str, str]:
    """Health check endpoint that verifies service readiness.

    Returns:
        Dict[str, str]: Simple service and S3 health status.
    """
    s3_status = "connected" if "s3_client" in resources else "disconnected"
    return {
        "status": "healthy",
        "message": "Data-prep API is operational",
        "s3_connection": s3_status,
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent
    logging_config_path = os.path.join(
        project_root,
        "conf",
        "logging.yaml",
    )

    with open(logging_config_path, encoding="utf-8") as file:
        log_config = yaml.safe_load(file)

    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_config=log_config)
