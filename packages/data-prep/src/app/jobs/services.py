# app/jobs/services.py
import asyncio
import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import HTTPException, status

from ..core.config import settings
from ..core.s3_client import get_s3_client
from .schemas import (
    DataPrepRequest,
    DataPrepRequestResponse,
    JobInfo,
    JobStatus,
)
from .tasks import run_data_processing


class JobService:
    def __init__(self):
        """
        Initialize the JobService.

        Sets up:
          - Cached S3 client and target bucket
          - Async queue for job IDs
          - Dictionaries for job status and input metadata
        """
        # S3 client & bucket
        self.s3_client = get_s3_client()
        self.bucket = settings.AWS_S3_BUCKET

        # in-memory job state
        self.queue = asyncio.Queue()
        self.statuses: Dict[str, JobInfo] = {}
        self.inputs: Dict[str, Dict] = {}  # maps run_id → {"path":..., "overrides":...}
        self.job_running = False

    async def create_job(self, req: DataPrepRequest) -> DataPrepRequestResponse:
        """
        Enqueue a new data preparation job.

        Args:
            req (DataPrepRequest): Contains raw chat data and optional overrides.

        Returns:
            DataPrepRequestResponse: Includes run_id, status, and a message.

        Raises:
            HTTPException 400: If `chats` payload is not a list or dict.
            HTTPException 409: If a UUID collision occurs (extremely unlikely).
        """
        # 1) Generate ID and collision check (very unlikely)
        run_id = uuid.uuid4().hex
        if run_id in self.statuses:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {run_id} already exists ({self.statuses[run_id].status})",
            )

        # 2) Validate payload
        chats = req.chats
        if not isinstance(chats, (list, dict)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`chats` must be a list or object",
            )

        # 3) Prepare overrides
        overrides = req.model_dump(exclude={"chats"}, exclude_none=True)

        # 4) Write raw chats to temp JSON
        temp_dir = Path(tempfile.gettempdir()) / "chat_data_prep"
        temp_dir.mkdir(parents=True, exist_ok=True)
        raw_path = temp_dir / f"{run_id}.json"
        raw_path.write_text(json.dumps(chats), encoding="utf-8")

        # 5) Store into inputs
        self.inputs[run_id] = {
            "path": str(raw_path),
            "overrides": overrides,
        }

        # 6) Record job status (with queue position)
        position = self.queue.qsize() + (1 if self.job_running else 0)
        job = JobInfo(
            run_id=run_id,
            status=JobStatus.QUEUED,
            position_in_queue=position,
            created_at=datetime.utcnow(),
        )
        self.statuses[run_id] = job

        # 7) Enqueue and return
        await self.queue.put(run_id)
        return DataPrepRequestResponse(
            run_id=run_id,
            status=job.status.name.lower(),
            message="Job accepted and enqueued",  # <-- add this
        )

    async def worker_loop(self):
        """
        Background worker loop that pulls jobs off the queue, processes them,
        and updates each job’s status when done (or failed).
        """
        while True:
            run_id = await self.queue.get()
            job = self.statuses[run_id]

            # mark running
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            self.job_running = True

            # update positions of queued jobs
            self._update_queue_positions()

            # grab the spec (path + overrides) and remove it
            spec = self.inputs.pop(run_id)

            try:
                # dispatch your processing function in a thread
                stats = await asyncio.to_thread(
                    run_data_processing,
                    run_id=run_id,
                    raw_json_path=spec["path"],
                    s3_client=self.s3_client,
                    s3_bucket_name=self.bucket,
                    overrides=spec["overrides"],
                )
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.stats = stats

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)

            finally:
                self.job_running = False
                self.queue.task_done()

    def get_status(self, run_id: str) -> JobInfo:
        """
        Fetch the JobInfo for a given run_id.

        Args:
            run_id (str): The identifier of the job.

        Returns:
            JobInfo: Status object, or None if not found.
        """
        return self.statuses.get(run_id)

    def _update_queue_positions(self):
        """
        Recompute `position_in_queue` for all queued jobs, in FIFO order.
        """
        queued = [
            rid
            for rid, info in self.statuses.items()
            if info.status == JobStatus.QUEUED
        ]
        for idx, rid in enumerate(queued):
            self.statuses[rid].position_in_queue = idx + 1


# singleton
job_service_singleton = JobService()
