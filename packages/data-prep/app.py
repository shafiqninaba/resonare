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

import uvicorn
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

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
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    run_id: str
    status: JobStatus
    position_in_queue: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class APIResponse(BaseModel):
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
    resources["s3_bucket"] = os.getenv("AWS_S3_BUCKET")
    resources["inputs"] = {}  # run_id → {"path": ...}
    logger.info(f"S3 client initialized, using bucket: {resources['s3_bucket']}")

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
    """Worker process that pulls jobs from queue and runs them one at a time"""
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
                logger.info(f"Job {run_id} completed successfully")
            except Exception as e:
                job_info.status = JobStatus.FAILED
                job_info.error = str(e)
                job_info.completed_at = datetime.now()
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


def update_queue_positions():
    """Update position in queue for all queued jobs"""
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
async def process_json(request: Request):
    """
    Start a data processing job with the specified run ID

    The job will be queued and processed when resources are available
    """
    run_id = uuid.uuid4().hex

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

    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise HTTPException(415, "Content‑Type must be application/json")

    try:
        body_bytes = await request.body()  # raw bytes (cached)
        data: Any = json.loads(body_bytes)
    except Exception:
        raise HTTPException(400, "Malformed JSON")

    # quick sanity‑check
    if not isinstance(data, (list, dict)):
        raise HTTPException(400, "Body must be a list or a Telegram export object")

    # write to OS temp dir
    tmp_root = Path(tempfile.gettempdir()) / "chat_data_prep"
    tmp_root.mkdir(exist_ok=True)
    raw_path = tmp_root / f"{run_id}.json"
    raw_path.write_bytes(body_bytes)

    resources["inputs"][run_id] = {"path": str(raw_path)}

    job_status[run_id] = JobInfo(
        run_id=run_id,
        status=JobStatus.QUEUED,
        position_in_queue=job_queue.qsize() + (1 if job_running else 0),
        created_at=datetime.now(),
    )
    await job_queue.put(run_id)

    return APIResponse(status="queued", message="JSON accepted", run_id=run_id)


@app.get("/data-prep/{run_id}/status")
async def status(run_id: str):
    if run_id not in job_status:
        raise HTTPException(404, "run_id not found")
    return job_status[run_id]


@app.get("/data-prep/queue")
async def queue():
    return {"running": job_running, "queue_size": job_queue.qsize(), "jobs": job_status}


@app.get("/data-prep/health")
async def health():
    ok = resources.get("s3_client") is not None
    return {"status": "healthy", "s3": "connected" if ok else "missing"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
