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
    def __init__(self):
        self.s3_client = get_s3_client()
        self.bucket = settings.AWS_S3_BUCKET

        # in-memory queue + state
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.statuses: Dict[str, JobInfo] = {}
        self.running = False

    async def submit(self, req: FineTuneRequest) -> FineTuneResponse:
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
        Background loop: pull run_ids off the queue,
        mark them running, call run_fine_tuning in a thread,
        then update status/result.
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
        return self.statuses.get(run_id)

    def _update_positions(self):
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
    return job_service_singleton
