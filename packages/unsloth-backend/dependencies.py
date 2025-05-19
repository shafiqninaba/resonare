import asyncio
import os
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime

import torch
from app.models import JobInfo, JobStatus
from fastapi import FastAPI, HTTPException
from src.fine_tune import run_fine_tuning
from src.general_utils import (
    downloadDirectoryFroms3,
    list_directories_in_bucket,
    setup_logger,
    setup_s3_client,
)
from unsloth import FastLanguageModel, FastModel

# Set up logger
logger = setup_logger("api")

# Dictionary to store connections and resources
resources = {}

# --- Fine-tuning job queue and tracking ---
job_queue = asyncio.Queue()
job_status = {}
job_running = False

# --- Inference Model Cache and Locks ---
# Cache to store loaded models: {run_id: (model, tokenizer)}
model_cache = {}
# Locks to prevent concurrent loading for the same run_id
# Use defaultdict to create locks on demand
loading_locks = defaultdict(asyncio.Lock)

# --- Global Inference Variables ---
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
S3_BUCKET = os.getenv("AWS_S3_BUCKET")


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

    # Clear model cache and free GPU memory
    logger.info("Clearing model cache")
    model_cache.clear()
    loading_locks.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")

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


async def load_or_get_model(run_id: str):
    """
    Loads a model if not in cache, otherwise returns the cached model.
    Uses locking to prevent concurrent loads of the same model.
    """
    # Check cache first (without lock for read access)
    if run_id in model_cache:
        logger.info(f"Using cached model for run_id: {run_id}")
        return model_cache[run_id]

    # Acquire the lock specific to this run_id
    async with loading_locks[run_id]:
        # Double-check cache after acquiring lock (another request might have loaded it)
        if run_id in model_cache:
            logger.info(
                f"Using cached model for run_id (loaded by concurrent request): {run_id}"
            )
            return model_cache[run_id]

        # --- Model not in cache, proceed to load ---
        logger.info(f"Cache miss. Loading model for run_id: {run_id}")
        if not S3_BUCKET:
            logger.error("AWS_S3_BUCKET environment variable not set.")
            raise HTTPException(
                status_code=500, detail="Server configuration error: S3 bucket not set."
            )

        # Create a temporary directory *only for downloading*
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory for download: {temp_dir}")
            dir_name = f"{run_id}/lora_model"

            try:
                # Check if the directory exists in S3
                directories = list_directories_in_bucket(S3_BUCKET)
                if run_id not in directories:
                    logger.error(
                        f"Directory {dir_name} not found in S3 bucket {S3_BUCKET}."
                    )
                    raise HTTPException(
                        status_code=404, detail=f"Model not found for run_id {run_id}"
                    )

                logger.info(
                    f"Directory {dir_name} found in S3 bucket {S3_BUCKET}. Proceeding to download."
                )
                # Download model files
                downloadDirectoryFroms3(S3_BUCKET, dir_name, temp_dir)
                logger.info(
                    f"Successfully downloaded model from s3://{S3_BUCKET}/{dir_name} to {temp_dir}"
                )

                # Load the model and tokenizer from the temporary directory
                logger.info(
                    f"Loading model and tokenizer from {temp_dir} into memory..."
                )
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=temp_dir,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=DTYPE,
                    load_in_4bit=LOAD_IN_4BIT,
                )
                FastLanguageModel.for_inference(model)  # Prepare for inference

                # Gemma Changes
                #     model, tokenizer = FastModel.from_pretrained(
                #     model_name = temp_dir,
                #     max_seq_length=MAX_SEQ_LENGTH,
                #     dtype=DTYPE,
                #     load_in_4bit=LOAD_IN_4BIT,
                # )
                #     FastModel.for_inference(model)  # Prepare for inference

                logger.info(
                    f"Model and tokenizer for run_id {run_id} loaded successfully."
                )

                # --- Store in cache ---
                model_cache[run_id] = (model, tokenizer)
                logger.info(f"Model for run_id {run_id} stored in cache.")

            except Exception as e:
                logger.error(
                    f"Failed to download or load model for run_id {run_id} from {temp_dir}: {e}",
                    exc_info=True,
                )
                # Ensure cache entry is not created on failure
                if run_id in model_cache:
                    del model_cache[run_id]
                raise HTTPException(
                    status_code=500, detail=f"Failed to load model for run_id {run_id}"
                )

        # Return the newly loaded model from cache
        return model_cache[run_id]
