import logging
import os
import tempfile
from typing import Dict, Tuple

import boto3
import torch
from transformers import PreTrainedTokenizer
from unsloth import FastLanguageModel  # or FastModel

logger = logging.getLogger(__name__)


def download_and_load_model(
    run_id: str, s3_client: boto3.client, bucket: str
) -> Tuple[torch.nn.Module, PreTrainedTokenizer, Dict[str, str]]:
    """
    1) Download the directory `{run_id}/lora_model` from S3 into a temp dir
    2) HeadObject on `{run_id}/data/train.jsonl` for metadata headers
    3) Load model & tokenizer from that temp dir
    4) Return (model, tokenizer, metadata_headers)
    """
    tmpdir = tempfile.mkdtemp()
    model_prefix = f"{run_id}/lora_model"
    local_dir = os.path.join(tmpdir, "lora_model")
    os.makedirs(local_dir, exist_ok=True)

    # list objects under that prefix, download each
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=model_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(model_prefix) + 1 :]
            if not rel:
                continue
            target = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            s3_client.download_file(bucket, key, target)

    # grab metadata from the data file
    data_key = f"{run_id}/data/train.jsonl"
    hdrs = s3_client.head_object(Bucket=bucket, Key=data_key)["ResponseMetadata"][
        "HTTPHeaders"
    ]

    # load model + tokenizer
    logger.info(f"Loading model from {local_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_dir,
        device_map="auto",
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer, hdrs


def generate_reply(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    messages: list[dict],
    temperature: float,
) -> str:
    """
    Tokenize the chat, run model.generate, and decode the assistant’s reply.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    output = model.generate(
        input_ids=inputs,
        temperature=temperature,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs.shape[1]
    reply = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
    return reply
