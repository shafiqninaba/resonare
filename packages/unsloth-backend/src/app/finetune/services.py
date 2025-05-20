import asyncio
from datetime import datetime
from typing import Dict

from fastapi import HTTPException, status

from ..core.config import settings
from ..core.s3_client import get_s3_client
from .schemas import (
    FineTuneRequest,
    FineTuneResponse,
    JobInfo,
    JobStatus,
)
from .tasks import run_fine_tuning


class JobService:
    """
    Manages the lifecycle of fine-tuning jobs: queuing, execution, and status tracking.

    Attributes:
        s3_client: Cached boto3 S3 client for interacting with the storage bucket.
        bucket: Name of the S3 bucket where training artifacts live.
        queue: In-memory FIFO queue of pending run IDs.
        statuses: Mapping from run_id to its JobInfo (status, timestamps, etc.).
        running: Flag indicating whether a job is currently in progress.
    """
    def __init__(self):
        """Initializes the JobService with S3 client, bucket name, and empty job state."""
        self.s3_client = get_s3_client()
        self.bucket = settings.AWS_S3_BUCKET

        # in-memory queue + state
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.statuses: Dict[str, JobInfo] = {}
        self.running = False

    async def submit(self, req: FineTuneRequest) -> FineTuneResponse:
        """
        Enqueue a new fine-tuning job.

        Args:
            req (FineTuneRequest): The request payload containing the run_id.

        Returns:
            FineTuneResponse: Confirmation that the job was queued, including its position.

        Raises:
            HTTPException: If a job with the same run_id already exists (409 Conflict).
        """
        run_id = req.run_id
        if run_id in self.statuses:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job {run_id} already exists",
            )

        # record new job
        position = self.queue.qsize() + (1 if self.running else 0)
        job = JobInfo(
            run_id=run_id,
            status=JobStatus.QUEUED,
            position_in_queue=position,
            created_at=datetime.utcnow(),
        )
        self.statuses[run_id] = job

        # enqueue and return
        await self.queue.put(run_id)
        return FineTuneResponse(
            run_id=run_id,
            status=job.status.value,
            message=f"Queued at position {position}",
        )

    async def worker_loop(self):
        """
        Background loop to process fine-tuning jobs one at a time.

        Workflow:
          1. Pull the next run_id from the queue.
          2. Mark its status as RUNNING and record the start time.
          3. Dispatch `run_fine_tuning` in a separate thread.
          4. On success, mark COMPLETED; on failure, mark FAILED with the error.
          5. Clear the `running` flag and mark the task as done.
        """
        while True:
            run_id = await self.queue.get()
            job = self.statuses[run_id]

            # mark running
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            self.running = True

            # update positions of queued jobs
            self._update_positions()

            try:
                # run the heavy lifting off the event loop
                await asyncio.to_thread(
                    run_fine_tuning,
                    run_id,
                    self.s3_client,
                    self.bucket,
                )
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)

            finally:
                self.running = False
                self.queue.task_done()

    def get_status(self, run_id: str) -> JobInfo:
        """
        Retrieve the current status of a given fine-tuning job.

        Args:
            run_id (str): The unique identifier of the job.

        Returns:
            JobInfo: The status record for the job, or None if not found.
        """
        return self.statuses.get(run_id)

    def _update_positions(self):
        """
        Recompute the `position_in_queue` for each queued job.

        Ensures that every JobInfo in QUEUED state has an accurate index.
        """
        queued = [
            rid
            for rid, info in self.statuses.items()
            if info.status == JobStatus.QUEUED
        ]
        for idx, rid in enumerate(queued):
            self.statuses[rid].position_in_queue = idx + 1


# singleton + DI helper
job_service_singleton = JobService()


def get_job_service() -> JobService:
    """
    Dependency injector for JobService singleton.

    Returns:
        JobService: The application-wide JobService instance.
    """
    return job_service_singleton
