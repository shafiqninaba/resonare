from fastapi import APIRouter, Depends

from .schemas import InferenceRequest, InferenceResponse
from .services import InferenceService, get_inference_service

router = APIRouter(prefix="/infer", tags=["inference"])


@router.post("", response_model=InferenceResponse)
async def infer(
    payload: InferenceRequest,
    svc: InferenceService = Depends(get_inference_service),
):
    """Run chat-style inference.

    This endpoint takes a conversation history and a run_id, ensures the
    corresponding model is loaded (with per-run_id locking and caching),
    and generates an assistant reply.

    Args:
        payload (InferenceRequest):  
            The request body containing:
                - run_id (str): Identifier of the fine-tuned model to use.
                - messages (List[Dict[str, str]]): Chat history messages.
                - temperature (Optional[float]): Sampling temperature.
        svc (InferenceService):  
            Dependency-injected service responsible for loading models
            and performing inference.

    Returns:
        InferenceResponse:  
            The generated assistant reply, including:
                - run_id (str): Echoed model identifier.
                - response (str): Generated text.
                - metadata (Dict[str, Any]): Model metadata from training.

    Raises:
        HTTPException:  
            If the payload is invalid, no model is found, or an internal
            error occurs during inference.
    """
    return await svc.infer(payload)
