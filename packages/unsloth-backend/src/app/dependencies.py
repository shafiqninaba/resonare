# src/app/dependencies.py
"""
Central dependency injection module.
Export your application dependencies for use with FastAPI's Depends().
"""
import boto3

from .core.config import settings
from .finetune.services import JobService as FineTuneJobService
from .inference.services import InferenceService
from .core.s3_client import get_s3_client
from .finetune.services import get_job_service as get_fine_tune_job_service
from .inference.services import get_inference_service


def get_settings_dep():
    """Returns the application Settings."""
    return settings


def get_s3_client_dep() -> boto3.client:
    """Returns a cached S3 client by calling the lru_cached function."""
    return get_s3_client()


def get_fine_tune_job_service_dep() -> FineTuneJobService:
    """Returns the singleton Finetune JobService instance."""
    return get_fine_tune_job_service()


def get_inference_service_dep() -> InferenceService:
    """Returns the singleton Inference JobService instance."""
    return get_inference_service()
