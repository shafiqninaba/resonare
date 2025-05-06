import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

import hydra
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.fine_tune import run_fine_tuning
from src.general_utils import setup_logger, setup_s3_client

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("fine_tuning_api")


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


# Dictionary to store connections and resources
resources = {}

# Create a job queue and job status tracking
job_queue = asyncio.Queue()
job_status: Dict[str, JobInfo] = {}
job_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup resources on startup
    logger.info("Setting up S3 client on application startup")
    resources["s3_client"] = setup_s3_client()
    resources["s3_bucket"] = os.getenv("AWS_S3_BUCKET")
    logger.info(f"S3 client initialized, using bucket: {resources['s3_bucket']}")

    # Start the job processor
    job_processor_task = asyncio.create_task(process_job_queue())
    resources["job_processor"] = job_processor_task
    logger.info("Job processor started")

    yield

    # Clean up resources on shutdown
    if "job_processor" in resources:
        resources["job_processor"].cancel()
        logger.info("Job processor cancelled")

    logger.info("Cleaning up resources")
    resources.clear()
    logger.info("Resources cleaned up")


async def process_job_queue():
    """Worker process that pulls jobs from queue and runs them one at a time"""
    global job_running

    while True:
        try:
            run_id = await job_queue.get()
            job_info = job_status[run_id]
            job_info.status = JobStatus.RUNNING
            job_info.position_in_queue = None
            job_info.started_at = datetime.now()
            job_running = True

            # Update queue positions for remaining jobs
            update_queue_positions()

            logger.info(f"Starting fine-tuning for job {run_id} from queue")

            # Run the fine-tuning task
            try:
                # Run synchronously to block the queue processing until complete
                await asyncio.to_thread(run_fine_tuning, run_id, resources)
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
    title="Model Fine-tuning API",
    description="API for fine-tuning language models",
    lifespan=lifespan,
)


class RunIDRequest(BaseModel):
    run_id: str


class TrainingResponse(BaseModel):
    status: str
    message: str
    run_id: str


@app.post("/fine-tune", response_model=TrainingResponse)
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


@app.get("/health")
async def health_check():
    """Check if the API is running"""
    s3_status = "connected" if "s3_client" in resources else "disconnected"
    return {
        "status": "healthy",
        "message": "Fine-tuning API is operational",
        "s3_connection": s3_status,
    }


@app.get("/fine-tune/{run_id}/status")
async def get_job_status(run_id: str):
    """Get the status of a specific fine-tuning job"""
    if run_id not in job_status:
        raise HTTPException(
            status_code=404, detail=f"Job with run_id {run_id} not found"
        )

    return job_status[run_id]


@app.get("/fine-tune/queue")
async def get_queue_status():
    """Get the status of all jobs in the queue"""
    return {"running": job_running, "queue_size": job_queue.qsize(), "jobs": job_status}


if __name__ == "__main__":
    try:
        with hydra.initialize(config_path="conf"):
            cfg = hydra.compose(config_name="config")

        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        uvicorn.run("main:app", host="0.0.0.0", port=8000, log_config=logging_config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        uvicorn.run("main:app", host="0.0.0.0", port=8000)
