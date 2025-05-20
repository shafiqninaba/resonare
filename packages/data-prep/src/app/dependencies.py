# app/dependencies.py
"""
Central dependency injection module.
Export your application dependencies for use with FastAPI's Depends().
"""

# This is the settings instance from config.py (decorated with lru_cache)
from app.core.config import settings

# This is the function decorated with lru_cache
from app.core.s3_client import (
    get_s3_client,
)

# This is the JobService singleton instance
from app.jobs.services import (
    job_service_singleton,
)


def get_settings_dep():
    """Returns the application Settings."""
    return settings


def get_s3_client_dep():
    """Returns a cached S3 client by calling the lru_cached function."""
    return get_s3_client()


def get_job_service_dep():
    """Returns the singleton JobService instance."""
    return job_service_singleton
