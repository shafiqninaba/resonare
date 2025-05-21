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
    """
    Service responsible for loading fine-tuned models from S3 (with per-run_id caching
    and asyncio locks) and performing chat-style inference.

    Attributes:
        _s3 (boto3.client): Cached S3 client.
        _bucket (str): Name of the S3 bucket where models are stored.
        _model_cache (Dict[str, _ModelTokMeta]): Maps run_id to (model, tokenizer, metadata).
        _locks (defaultdict[str, asyncio.Lock]): Per-run_id locks to prevent concurrent loads.
    """

    def __init__(self):
        """
        Initialize the InferenceService by setting up S3 access and in-memory caches.
        """
        self._s3: boto3.client = get_s3_client()
        self._bucket: str = settings.AWS_S3_BUCKET

        # caches and per-run_id locks
        self._model_cache: Dict[str, _ModelTokMeta] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def _ensure_loaded(self, run_id: str) -> _ModelTokMeta:
        """
        Ensure that the model, tokenizer, and metadata for the given run_id
        are loaded into the cache. If not already cached, download from S3
        and load them, using an asyncio lock to prevent duplicate work.

        Args:
            run_id (str): Hex UUID identifying the fine-tuned model.

        Returns:
            Tuple[model, tokenizer, metadata]:  
                - model: Loaded model object.  
                - tokenizer: Corresponding tokenizer.  
                - metadata (Dict[str, str]): Parsed training metadata headers.

        Raises:
            HTTPException:  
                - 404 if `download_and_load_model` raises a not found error.  
                - 500 for any other failure during download or load.
        """
        # Fast path
        if run_id in self._model_cache:
            return self._model_cache[run_id]

        lock = self._locks[run_id]
        async with lock:
            # Double-check after acquiring lock
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
                # propagate 404 / client errors
                raise
            except Exception as e:
                # wrap unexpected errors
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load model {run_id}: {e}",
                )
            self._model_cache[run_id] = (model, tokenizer, meta)
            return model, tokenizer, meta

    async def infer(self, req: InferenceRequest) -> InferenceResponse:
        """
        Perform chat-style inference using a fine-tuned model.

        Workflow:
          1. Validate that `messages` is non-empty.
          2. Ensure the model/tokenizer/meta for `run_id` are loaded.
          3. Generate a response via `generate_reply`.

        Args:
            req (InferenceRequest):  
                - run_id: ID of the fine-tuned model.  
                - messages: Conversation history.  
                - temperature: Sampling temperature.

        Returns:
            InferenceResponse:  
                - run_id: Echoed model ID.  
                - response: Generated assistant reply.  
                - metadata: Training metadata headers from S3.

        Raises:
            HTTPException:  
                - 400 if `messages` is empty.  
                - 404 or 500 from `_ensure_loaded` if model cannot be loaded.
        """
        run_id = req.run_id
        if not req.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="`messages` must be a non-empty list",
            )

        model, tokenizer, meta = await self._ensure_loaded(run_id)
        reply = generate_reply(model, tokenizer, req.messages, req.temperature)
        return InferenceResponse(run_id=run_id, response=reply, metadata=meta)


# singleton + DI helper
_inference_service = InferenceService()


def get_inference_service() -> InferenceService:
    """
    Dependency injector function for FastAPI.

    Returns:
        InferenceService: The shared singleton instance.
    """
    return _inference_service
