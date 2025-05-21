from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """
    Request payload for running chat-style inference.

    Attributes:
        run_id (str):  
            Hex UUID identifying the fine-tuned model in S3.  
            Must match the pattern `^[0-9a-f]{32}$`.
        messages (List[Dict[str, str]]):  
            Conversation history, e.g.  
            `[{'role':'user','content':'Hi'}, {'role':'assistant','content':'…'}]`.
        temperature (Optional[float]):  
            Sampling temperature for generation.  
            Must be between 0.0 and 5.0. Defaults to 1.0.
    """

    run_id: str = Field(
        ...,
        pattern=r"^[0-9a-f]{32}$",
        description="Hex UUID identifying the fine-tuned model in S3",
    )
    messages: List[Dict[str, str]] = Field(
        ...,
        description=(
            "Conversation history, e.g. "
            "[{'role':'user','content':'Hi'}, {'role':'assistant','content':'…'}]"
        ),
    )
    temperature: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=5.0,
        description="Sampling temperature for generation (0.0–5.0)",
    )


class InferenceResponse(BaseModel):
    """
    Response returned by the /infer endpoint.

    Attributes:
        run_id (str):  
            Echoed identifier of the model used for inference.
        response (str):  
            Generated assistant reply.
        metadata (Dict[str, str]):  
            Metadata extracted from the model files (e.g., training headers).
    """

    run_id: str = Field(..., description="Identifier of the model used")
    response: str = Field(..., description="Generated assistant reply")
    metadata: Dict[str, str] = Field(
        {},
        description="Metadata from the model files (e.g. training headers)",
    )
