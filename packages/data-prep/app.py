#!/usr/bin/env python3
"""
Chat‑Data‑Prep API
==================
POST /data-prep/process      ── client sends ONE merged JSON payload (application/json)
GET  /data-prep/{run_id}/status
GET  /data-prep/queue
GET  /health
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import uvicorn
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import yaml

from enum import Enum
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
                await asyncio.to_thread(run_data_processing, run_id, resources, spec)
                job_info.status = JobStatus.COMPLETED
                job_info.completed_at = datetime.now()
                logger.info(f"Data processing for job {run_id} completed successfully")

                # Post-processing trigger
                logger.info(f"Triggering finetuning for job {run_id}")
                await finetuning_hook(
                    endpoint="http://localhost:8020/fine-tune", run_id=run_id
                )

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


@app.post("/data-prep/process", response_model=APIResponse)
async def submit_data_prep_job(request: Request) -> APIResponse:
    """Receives a JSON chat export and queues a data preprocessing job.

    The request must be a single JSON payload (Content-Type: application/json)
    that is either:
      - a list of chat objects, or
      - a Telegram-style export object with chats inside.

    The raw request is written to a temporary file and passed to the background
    worker via an async job queue.

    Args:
        request (Request): Incoming HTTP request with the JSON chat export.

    Returns:
        APIResponse: Object containing run_id and queue status.

    Raises:
        HTTPException: On content type errors, malformed JSON, or bad structure.
    """
    # Generate a new unique job ID
    run_id = uuid.uuid4().hex

    # Check if S3 client is available (setup during app lifespan)
    if "s3_client" not in resources:
        raise HTTPException(
            status_code=500,
            detail="S3 client not initialized. Required for uploads.",
        )

    # Check for duplicate job ID (highly unlikely with uuid4, but safe)
    if run_id in job_status:
        raise HTTPException(
            status_code=409,
            detail=f"Job with run_id {run_id} already exists with status {job_status[run_id].status}",
        )

    # Validate content type header
    content_type = request.headers.get("content-type", "").split(";")[0]
    if content_type != "application/json":
        raise HTTPException(
            status_code=415,
            detail="Content-Type must be application/json",
        )

    # Parse JSON body (raw bytes for storage, parsed for validation)
    try:
        body_bytes = await request.body()
        parsed_data: Any = json.loads(body_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed JSON")

    # Sanity check: we expect either a list of chats or a Telegram export object
    if not isinstance(parsed_data, (list, dict)):
        raise HTTPException(
            status_code=400,
            detail="Body must be a list of chats or a Telegram export object",
        )

    # Write raw request to a temp file for processing
    temp_dir = Path(tempfile.gettempdir()) / "chat_data_prep"
    temp_dir.mkdir(parents=True, exist_ok=True)

    raw_path = temp_dir / f"{run_id}.json"
    raw_path.write_bytes(body_bytes)

    # Store path for the worker to access
    resources["inputs"][run_id] = {"path": str(raw_path)}

    # Register job metadata
    job_status[run_id] = JobInfo(
        run_id=run_id,
        status=JobStatus.QUEUED,
        position_in_queue=job_queue.qsize() + (1 if job_running else 0),
        created_at=datetime.now(),
    )

    # Enqueue the job
    await job_queue.put(run_id)

    return APIResponse(
        status="queued",
        message="Job accepted and enqueued",
        run_id=run_id,
    )


@app.get("/data-prep/jobs")
async def get_all_job_status() -> Dict[str, JobInfo]:
    """Return the full mapping of all job statuses.

    Returns:
        Dict[str, JobInfo]: All known jobs, regardless of status.
    """
    return job_status


@app.get("/data-prep/jobs/{run_id}")
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
        raise HTTPException(404, "Job not found")
    return job_status[run_id]


@app.get("/data-prep/system/queue")
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


@app.get("/data-prep/system/health")
async def get_health_status() -> Dict[str, str]:
    """Health check endpoint that verifies service readiness.

    Returns:
        Dict[str, str]: Simple service and S3 health status.
    """
    healthy = resources.get("s3_client") is not None
    return {
        "status": "healthy",
        "s3": "connected" if healthy else "missing",
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

    uvicorn.run(
        "app:app", host="0.0.0.0", port=8000, reload=True, log_config=log_config
    )
