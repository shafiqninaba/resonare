# app/main.py
import asyncio
import os
from logging import getLogger
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI

from app.core.config import settings
from app.core.s3_client import get_s3_client
from app.dependencies import get_job_service_dep
from app.modules.jobs.router import router as jobs_router
from app.modules.system.router import router as system_router
from app.utils.general import setup_standard_logging

# Initialize logging
project_root = Path(__file__).parent
logger = getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_standard_logging(
    logging_config_path=os.path.join(
        project_root,
        "conf",
        "logging.yaml",
    ),
)
logger.info("Logging configuration set up successfully.")


async def lifespan(app: FastAPI):
    # Eagerly initialize & validate S3 client
    logger.info("Setting up S3 client on application startup")
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=settings.AWS_S3_BUCKET)
    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(f"Failed to connect to S3: {e}")
    logger.info("S3 client initialized successfully.")

    # Start background job worker
    logger.info("Setting up the job queue...")
    job_service = get_job_service_dep()
    worker_task = asyncio.create_task(job_service.worker_loop())

    yield  # application is up

    # Shutdown: cancel the worker]
    logger.info("Cleaning up resources")
    worker_task.cancel()
    logger.info("Resources cleaned up")


# Create the FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    lifespan=lifespan,
)

# Include routers
app.include_router(system_router)
app.include_router(jobs_router)
