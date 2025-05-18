import torch
from fastapi import APIRouter, HTTPException
from typing import Dict

from app.dependencies import load_or_get_model, logger
from app.models import InferenceRequest

router = APIRouter(tags=["inference"])


@router.post("/infer")
async def run_inference(request_data: InferenceRequest):
    """Runs inference using a cached model specified by run_id, using conversation history."""
    run_id = request_data.run_id
    messages = request_data.messages  # Get the list of messages from the request

    # Basic validation: Ensure messages list is not empty
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    try:
        # Get model and tokenizer (loads from S3 and caches if not already loaded)
        model, tokenizer = await load_or_get_model(run_id)

        logger.info(
            f"Running inference for run_id: {run_id} with {len(messages)} messages..."
        )

        # Use the received message history directly
        inputs = tokenizer.apply_chat_template(
            messages,  # Pass the whole conversation
            tokenize=True,
            add_generation_prompt=True,  # Add prompt for the assistant to respond
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate the response
        # Consider making generation parameters configurable via request_data if needed
        output_tokens = model.generate(
            input_ids=inputs,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
            pad_token_id=tokenizer.eos_token_id,
            # You might need to adjust max_new_tokens if context gets long
            max_new_tokens=64,  # Example: Limit generated tokens
        )

        input_length = inputs.shape[1]
        decoded_output = tokenizer.decode(
            output_tokens[0, input_length:], skip_special_tokens=True
        )
         
        # Gemma Changes
    #     text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt = True, # Must add for generation
    # )
    #     outputs = model.generate(
    #         **tokenizer([text], return_tensors = "pt").to("cuda"),
    #         max_new_tokens = 64, # Increase for longer outputs!
    #         # Recommended Gemma-3 settings!
    #         temperature = 1.0, top_p = 0.95, top_k = 64,
    #     )
    #     decoded_output = tokenizer.batch_decode(outputs)
    #     decoded_output = decoded_output[0]
    
        logger.info(f"Inference successful for run_id: {run_id}")
        return {"run_id": run_id, "response": decoded_output}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like model loading failures)
        raise http_exc
    except Exception as e:
        logger.error(
            f"An error occurred during inference for run_id {run_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Internal server error during inference: {str(e)}"
        )