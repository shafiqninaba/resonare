from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataPrepRequest(BaseModel):
    """Standard request for preprocessing a list of chat messages and configuring fine-tuning.

    Attributes:
        chats (List[Dict[str, Any]]): List of chat dictionaries to be preprocessed.
        --------------------------------------------------------------------
        target_name (Optional[str]): Name under which to store the processed chat.
        system_prompt (Optional[str]): System message to prepend to every block.
        date_limit (Optional[str]): ISO date string; ignore messages after this date.
        convo_block_threshold_secs (int): Max gap in seconds before starting a new block.
        min_tokens_per_block (int): Minimum tokens per block.
        max_tokens_per_block (int): Maximum tokens per block.
        message_delimiter (str): Prefix for merged lines.
        --------------------------------------------------------------------
        name (Optional[str]): Model ID for fine-tuning.
        chat_template (Optional[str]): Template format for chat messages.
        r (Optional[int]): LoRA rank parameter.
        alpha (Optional[int]): LoRA alpha parameter.
        per_device_train_batch_size (Optional[int]): Batch size for training.
        gradient_accumulation_steps (Optional[int]): Gradient accumulation steps.
        warmup_steps (Optional[int]): Number of warmup steps.
        max_steps (Optional[int]): Maximum number of training steps.
    """

    # required payload
    chats: List[Dict[str, Any]] = Field(
        ..., description="List of chat dictionaries for preprocessing"
    )

    # --------------------------------------------------------------------
    # Data Preparation settings
    # --------------------------------------------------------------------
    target_name: Optional[str] = Field(
        None, description="Name under which to store processed chat"
    )
    system_prompt: Optional[str] = Field(
        None, description="System message to prepend to every block"
    )
    date_limit: Optional[str] = Field(
        None, description="ISO date string; ignore messages after this date"
    )
    convo_block_threshold_secs: int = Field(
        3600,
        description="Max gap in seconds before starting a new block",
    )
    min_tokens_per_block: int = Field(300, description="Minimum tokens per block")
    max_tokens_per_block: int = Field(800, description="Maximum tokens per block")
    message_delimiter: str = Field(">>>", description="Prefix for merged lines")

    # --------------------------------------------------------------------
    # Fine-tuning settings
    # --------------------------------------------------------------------
    name: Optional[str] = Field(None, description="Model ID for fine-tuning")
    chat_template: Optional[str] = Field(
        None, description="Template format for chat messages"
    )
    r: Optional[int] = Field(None, description="LoRA rank parameter")
    alpha: Optional[int] = Field(None, description="LoRA alpha parameter")
    per_device_train_batch_size: Optional[int] = Field(
        None, description="Batch size for training"
    )
    gradient_accumulation_steps: Optional[int] = Field(
        None, description="Gradient accumulation steps"
    )
    warmup_steps: Optional[int] = Field(None, description="Number of warmup steps")
    max_steps: Optional[int] = Field(
        None, description="Maximum number of training steps"
    )


class DataPrepRequestResponse(BaseModel):
    """Standard response returned after submitting a preprocessing job.

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
    """Metadata and lifecycle information for a single preprocessing job.

    Attributes:
        run_id (str): Unique identifier for the job.
        status (JobStatus): Current status of the job (e.g. queued, running).
        position_in_queue (Optional[int]): Position of the job in the queue, if applicable.
        created_at (datetime): Timestamp when the job was submitted.
        started_at (Optional[datetime]): Timestamp when processing started.
        completed_at (Optional[datetime]): Timestamp when processing completed or failed.
        error (Optional[str]): Error message if the job failed.
        stats (Optional[Dict[str, Any]]): Additional statistics or metadata about the chat statistics.
    """

    run_id: str
    status: JobStatus
    position_in_queue: Optional[int] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None
