import asyncio
from collections import defaultdict
from typing import Dict, Tuple

import boto3
from fastapi import HTTPException, status

from ..core.config import settings
from ..core.s3_client import get_s3_client
from .schemas import InferenceRequest, InferenceResponse
from .tasks import download_and_load_model, generate_reply

# type aliases for clarity
_ModelTokMeta = Tuple[any, any, Dict[str, str]]


class InferenceService:
    def __init__(self):
        self._s3: boto3.client = get_s3_client()
        self._bucket: str = settings.AWS_S3_BUCKET

        # caches and per-run_id locks
        self._model_cache: Dict[str, _ModelTokMeta] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def _ensure_loaded(self, run_id: str) -> _ModelTokMeta:
        # Fast path:
        if run_id in self._model_cache:
            return self._model_cache[run_id]

        # lock per run_id
        lock = self._locks[run_id]
        async with lock:
            # double-check
            if run_id in self._model_cache:
                return self._model_cache[run_id]

            try:
                model, tokenizer, meta = await asyncio.to_thread(
                    download_and_load_model,
                    run_id,
                    self._s3,
                    self._bucket,
                )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load model {run_id}: {e}",
                )
            self._model_cache[run_id] = (model, tokenizer, meta)
            return model, tokenizer, meta

    async def infer(self, req: InferenceRequest) -> InferenceResponse:
        run_id = req.run_id
        if not req.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`messages` must be non-empty list",
            )

        model, tokenizer, meta = await self._ensure_loaded(run_id)
        # generation
        reply = generate_reply(model, tokenizer, req.messages, req.temperature)
        return InferenceResponse(run_id=run_id, response=reply, metadata=meta)


# singleton + DI
_inference_service = InferenceService()


def get_inference_service() -> InferenceService:
    return _inference_service
