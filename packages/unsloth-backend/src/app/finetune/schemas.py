from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class FineTuneRequest(BaseModel):
    """
    Request payload to start a fine-tuning job.
    """

    run_id: str


class FineTuneResponse(BaseModel):
    """Standard response returned after submitting a fine-tuning job.

    Attributes:
        status (str): Status of the request (e.g. "queued").
        message (str): Human-readable message for client feedback.
        run_id (str): Unique identifier for the submitted job.
    """

    status: str
    message: str
    run_id: str


class JobStatus(str, Enum):
    """Enum representing the current status of a job in the queue."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Metadata and lifecycle information for a single finetuning job.

    Attributes:
        run_id (str): Unique identifier for the fine-tuning job.
        status (JobStatus): Current status of the fine-tuning job (e.g. queued, running).
        position_in_queue (Optional[int]): Position of the fine-tuning job in the queue, if applicable.
        created_at (datetime): Timestamp when the fine-tuning job was submitted.
        started_at (Optional[datetime]): Timestamp when fine-tuning started.
        completed_at (Optional[datetime]): Timestamp when fine-tuning completed or failed.
        error (Optional[str]): Error message if the fine-tuning job failed.

    """

    run_id: str
    status: JobStatus
    position_in_queue: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
