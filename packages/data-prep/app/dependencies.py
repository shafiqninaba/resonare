# app/dependencies.py
"""
Central dependency injection module.
Export your application dependencies for use with FastAPI's Depends().
"""

from app.core.config import settings
from app.core.s3_client import get_s3_client
from app.modules.jobs.services import job_service


def get_settings_dep():
    """Returns the application Settings."""
    return settings


def get_s3_client_dep():
    """Returns a cached S3 client."""
    return get_s3_client()


def get_job_service_dep():
    """Returns the singleton JobService."""
    return job_service
