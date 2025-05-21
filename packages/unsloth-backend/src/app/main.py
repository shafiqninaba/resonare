# src/app/main.py
import asyncio
from contextlib import asynccontextmanager

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI

# Core application components
from src.app.core.config import settings
from src.app.dependencies import get_fine_tune_job_service_dep, get_s3_client_dep
from src.app.finetune.router import router as finetune_router
from src.app.inference.router import router as inference_router

# Routers
from src.app.system.router import router as system_router
from src.app.utils.general import setup_logger

# Set up logging
logger = setup_logger(__name__)

# Load environment variables

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info(f"Starting up {settings.PROJECT_NAME}...")

    # Eagerly initialize & validate S3 client
    logger.info("Setting up S3 client on application startup")
    try:
        s3 = get_s3_client_dep()
        s3.head_bucket(Bucket=settings.AWS_S3_BUCKET)
    except (ClientError, BotoCoreError) as e:
        raise RuntimeError(f"Failed to connect to S3: {e}")
    logger.info("S3 client initialized successfully.")

    # 2. Start Fine-Tuning Job Worker
    logger.info("Starting background job worker...")
    job_service = get_fine_tune_job_service_dep()
    worker_task = asyncio.create_task(job_service.worker_loop())
    logger.info("Background job worker started.")

    # InferenceService is initialized on first use via DI, no startup action needed here
    # unless you want to pre-warm its cache or something similar.

    yield  # Application is ready

    # --- Shutdown ---
    logger.info("Shutting down background job worker...")  # Renamed log message
    worker_task.cancel()
    try:
        await worker_task  # Important to await for graceful cancellation
    except asyncio.CancelledError:
        logger.info("Worker task cancelled successfully.")
    logger.info("Background job worker shut down.")


# Create FastAPI app instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=getattr(settings, "API_VERSION_STR", "0.1.0"),
    lifespan=lifespan,
)

# Include Routers
app.include_router(system_router, tags=["System"])
app.include_router(finetune_router, tags=["Fine-Tuning"])
app.include_router(inference_router, tags=["Inference"])


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
