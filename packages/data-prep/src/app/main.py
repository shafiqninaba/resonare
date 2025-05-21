# app/main.py
import asyncio
import os
from logging import getLogger
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI

from .core.config import settings
from .dependencies import get_job_service_dep, get_s3_client_dep
from .jobs.router import router as jobs_router
from .system.router import router as system_router
from .utils.general import setup_standard_logging

# Initialize logging
project_root = Path(__file__).parent
logger = getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_standard_logging(
    logging_config_path=os.path.join(
        "/app/conf",
        "logging.yaml",
    ),
)
logger.info("Logging configuration set up successfully.")


async def lifespan(app: FastAPI):
    # Eagerly initialize & validate S3 client
    logger.info("Setting up S3 client on application startup")
    try:
        s3 = get_s3_client_dep()
        s3.head_bucket(Bucket=settings.AWS_S3_BUCKET)
    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(f"Failed to connect to S3: {e}")
    logger.info("S3 client initialized successfully.")

    # Start background job worker
    logger.info("Starting background job worker...")
    job_service = get_job_service_dep()
    worker_task = asyncio.create_task(job_service.worker_loop())
    logger.info("Background job worker started.")

    yield  # application is up

    # Shutdown: cancel the worker
    logger.info("Shutting down background job worker...")  # Renamed log message
    worker_task.cancel()
    try:
        await worker_task  # Important to await for graceful cancellation
    except asyncio.CancelledError:
        logger.info("Worker task cancelled successfully.")
    logger.info("Background job worker shut down.")


# Create the FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    lifespan=lifespan,
)

# Include routers
app.include_router(system_router)
app.include_router(jobs_router)


# Basic root endpoint
@app.get("/", tags=["Root"], include_in_schema=False)
async def read_root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}!",
        "status": "API is running",
    }


logger.info(
    f"{settings.PROJECT_NAME} initialized successfully. Listening for requests."
)
