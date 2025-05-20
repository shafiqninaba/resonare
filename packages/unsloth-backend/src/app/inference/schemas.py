from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """
    Request payload for running chat-style inference.
    """

    run_id: str = Field(
        ...,
        pattern=r"^[0-9a-f]{32}$",
        description="Hex UUID identifying the fine-tuned model in S3",
    )
    messages: List[Dict[str, str]] = Field(
        ...,
        description="Conversation history, e.g. "
        "[{'role':'user','content':'Hi'}, {'role':'assistant','content':'…'}]",
    )
    temperature: Optional[float] = Field(
        1.0, ge=0.0, le=5.0, description="Sampling temperature"
    )


class InferenceResponse(BaseModel):
    """
    Response returned by the /infer endpoint.
    """

    run_id: str
    response: str = Field(..., description="Generated assistant reply")
    metadata: Dict[str, str] = Field(
        {},
        description="Any metadata pulled from the model files (e.g. training headers)",
    )
