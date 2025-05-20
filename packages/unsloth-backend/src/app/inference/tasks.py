import logging
import os
import tempfile
from typing import Dict, Tuple, List

import boto3
import torch
from transformers import PreTrainedTokenizer
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


def download_and_load_model(
    run_id: str,
    s3_client: boto3.client,
    bucket: str
) -> Tuple[torch.nn.Module, PreTrainedTokenizer, Dict[str, str]]:
    """
    Download and load a fine-tuned model from S3.

    This function will:
      1. Create a temporary directory.
      2. Download all files under the S3 prefix "{run_id}/lora_model" into it.
      3. Read metadata headers from the training data file at "{run_id}/data/train.jsonl".
      4. Load the model and tokenizer from the downloaded directory.
      5. Return the loaded model, tokenizer, and metadata headers.

    Args:
        run_id (str): Hexadecimal UUID of the job/model to load.
        s3_client (boto3.client): Authenticated S3 client.
        bucket (str): Name of the S3 bucket containing model artifacts.

    Returns:
        Tuple[torch.nn.Module, PreTrainedTokenizer, Dict[str, str]]: 
            - model: The loaded PyTorch model, ready for inference.
            - tokenizer: Corresponding tokenizer instance.
            - metadata_headers: HTTP headers from the train.jsonl `head_object` call.
    """
    tmpdir = tempfile.mkdtemp()
    model_prefix = f"{run_id}/lora_model"
    local_dir = os.path.join(tmpdir, "lora_model")
    os.makedirs(local_dir, exist_ok=True)

    # Download all model files
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=model_prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(model_prefix) + 1:]
            if not rel_path:
                continue
            target_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            s3_client.download_file(bucket, key, target_path)

    # Fetch training metadata headers
    data_key = f"{run_id}/data/train.jsonl"
    response = s3_client.head_object(Bucket=bucket, Key=data_key)
    metadata_headers = response["ResponseMetadata"]["HTTPHeaders"]

    # Load model and tokenizer
    logger.info(f"Loading model from {local_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_dir,
        device_map="auto",
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer, metadata_headers


def generate_reply(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    temperature: float
) -> str:
    """
    Generate a single assistant reply given conversation history.

    Steps:
      1. Tokenize the chat history with the tokenizer's chat template.
      2. Move inputs to GPU if available.
      3. Call `model.generate` with specified sampling temperature.
      4. Decode only the newly generated tokens (post-input prompt length).

    Args:
        model (torch.nn.Module): Loaded model for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer matching the model.
        messages (List[Dict[str, str]]): Chat history, each dict has 'role' and 'content'.
        temperature (float): Sampling temperature (higher => more random).

    Returns:
        str: Decoded assistant reply.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        input_ids=inputs,
        temperature=temperature,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_length = inputs.shape[1]
    reply = tokenizer.decode(
        outputs[0, prompt_length:],
        skip_special_tokens=True
    )
    return reply
