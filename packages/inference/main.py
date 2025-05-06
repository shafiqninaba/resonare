import asyncio
import os
import tempfile
from collections import defaultdict
from typing import Dict, List  # Import List and Dict for typing

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.general_utils import (
    downloadDirectoryFroms3,
    list_directories_in_bucket,
    setup_logger,
)
from unsloth import FastLanguageModel

# Load environment variables
load_dotenv()

# --- Global Variables ---
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
S3_BUCKET = os.getenv("AWS_S3_BUCKET")

# Set up logger
logger = setup_logger("inference")

# --- Model Cache and Locks ---
# Cache to store loaded models: {run_id: (model, tokenizer)}
model_cache = {}
# Locks to prevent concurrent loading for the same run_id
# Use defaultdict to create locks on demand
loading_locks = defaultdict(asyncio.Lock)

# --- FastAPI App ---
app = FastAPI()


# --- Pydantic Model for Request Body ---
class InferenceRequest(BaseModel):
    run_id: str
    # Expect a list of message dictionaries instead of a single prompt
    messages: List[
        Dict[str, str]
    ]  # e.g., [{"role": "user", "content": "..."}, {"role": "assistant", ...}]


# --- Helper Function to Load Model (with Caching and Locking) ---
async def load_or_get_model(run_id: str):
    """
    Loads a model if not in cache, otherwise returns the cached model.
    Uses locking to prevent concurrent loads of the same model.
    """
    # ... (load_or_get_model function remains exactly the same) ...
    # Check cache first (without lock for read access)
    if run_id in model_cache:
        logger.info(f"Using cached model for run_id: {run_id}")
        return model_cache[run_id]

    # Acquire the lock specific to this run_id
    async with loading_locks[run_id]:
        # Double-check cache after acquiring lock (another request might have loaded it)
        if run_id in model_cache:
            logger.info(
                f"Using cached model for run_id (loaded by concurrent request): {run_id}"
            )
            return model_cache[run_id]

        # --- Model not in cache, proceed to load ---
        logger.info(f"Cache miss. Loading model for run_id: {run_id}")
        if not S3_BUCKET:
            logger.error("AWS_S3_BUCKET environment variable not set.")
            raise HTTPException(
                status_code=500, detail="Server configuration error: S3 bucket not set."
            )

        # Create a temporary directory *only for downloading*
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory for download: {temp_dir}")
            dir_name = f"{run_id}/models/lora_model"

            try:
                # Check if the directory exists in S3
                directories = list_directories_in_bucket(S3_BUCKET)
                if run_id not in directories:
                    logger.error(
                        f"Directory {dir_name} not found in S3 bucket {S3_BUCKET}."
                    )
                    raise HTTPException(
                        status_code=404, detail=f"Model not found for run_id {run_id}"
                    )

                logger.info(
                    f"Directory {dir_name} found in S3 bucket {S3_BUCKET}. Proceeding to download."
                )
                # Download model files
                downloadDirectoryFroms3(S3_BUCKET, dir_name, temp_dir)
                logger.info(
                    f"Successfully downloaded model from s3://{S3_BUCKET}/{dir_name} to {temp_dir}"
                )

                # Load the model and tokenizer from the temporary directory
                logger.info(
                    f"Loading model and tokenizer from {temp_dir} into memory..."
                )
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=temp_dir,
                    max_seq_length=MAX_SEQ_LENGTH,
                    dtype=DTYPE,
                    load_in_4bit=LOAD_IN_4BIT,
                )
                FastLanguageModel.for_inference(model)  # Prepare for inference
                logger.info(
                    f"Model and tokenizer for run_id {run_id} loaded successfully."
                )

                # --- Store in cache ---
                model_cache[run_id] = (model, tokenizer)
                logger.info(f"Model for run_id {run_id} stored in cache.")

            except Exception as e:
                logger.error(
                    f"Failed to download or load model for run_id {run_id} from {temp_dir}: {e}",
                    exc_info=True,
                )
                # Ensure cache entry is not created on failure
                if run_id in model_cache:
                    del model_cache[run_id]
                raise HTTPException(
                    status_code=500, detail=f"Failed to load model for run_id {run_id}"
                )
            # Temporary directory `temp_dir` is automatically cleaned up here by the `with` statement

        # Return the newly loaded model from cache
        return model_cache[run_id]


# --- API Endpoint ---
@app.post("/infer")
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
            max_new_tokens=512,  # Example: Limit generated tokens
        )

        input_length = inputs.shape[1]
        decoded_output = tokenizer.decode(
            output_tokens[0, input_length:], skip_special_tokens=True
        )

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


# --- Cleanup on Shutdown (Optional but Recommended) ---
@app.on_event("shutdown")
def shutdown_event():
    # ... (shutdown_event remains the same) ...
    global model_cache, loading_locks
    logger.info("FastAPI application shutting down. Clearing model cache.")
    # Explicitly clear cache and locks
    model_cache.clear()
    loading_locks.clear()
    # Potentially add GPU memory cleanup if necessary, though Python's GC and libraries often handle it.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared.")


# --- How to Run (Example using uvicorn) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
