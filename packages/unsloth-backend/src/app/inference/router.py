from fastapi import APIRouter, Depends

from .schemas import InferenceRequest, InferenceResponse
from .services import InferenceService, get_inference_service

router = APIRouter(prefix="/infer", tags=["inference"])


@router.post("", response_model=InferenceResponse)
async def infer(
    payload: InferenceRequest,
    svc: InferenceService = Depends(get_inference_service),
):
    """
    Run chat-style inference:
      - load or fetch the model (cached + locked per run_id)
      - generate the assistant reply
    """
    return await svc.infer(payload)
