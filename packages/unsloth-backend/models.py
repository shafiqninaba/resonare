from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


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


class RunIDRequest(BaseModel):
    run_id: str


class TrainingResponse(BaseModel):
    status: str
    message: str
    run_id: str


# --- Inference Models ---
class InferenceRequest(BaseModel):
    run_id: str
    # Expect a list of message dictionaries instead of a single prompt
    messages: List[
        Dict[str, str]
    ]  # e.g., [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
    temperature: Optional[float] = 1.5
