import os
import tempfile
import ast
from typing import Dict

import hydra
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from src.general_utils import setup_logger, upload_directory_to_s3
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    train_on_responses_only,
)
from trl import SFTTrainer

load_dotenv()

logger = setup_logger("fine_tuning")


def parse_metadata(headers):
    """Parse S3 metadata headers and convert to appropriate types"""
    cfg = {}
    for key, value in headers.items():
        if key.startswith('x-amz-meta-'):
            clean_key = key
            
            # Handle booleans
            if value.lower() in ('true', 'false'):
                cfg[clean_key] = value.lower() == 'true'
            
            # Handle integers
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                cfg[clean_key] = int(value)
            
            # Handle floats
            elif value.replace('.', '', 1).isdigit() or (value.startswith('-') and value[1:].replace('.', '', 1).isdigit()):
                cfg[clean_key] = float(value)
            
            # Handle lists
            elif value.startswith('[') and value.endswith(']'):
                try:
                    cfg[clean_key] = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    # Fallback to string if parsing fails
                    cfg[clean_key] = value
            
            # Keep as string for other values
            else:
                cfg[clean_key] = value
    
    return cfg


def run_fine_tuning(run_id: str, resources: Dict[str, any]) -> None:
    """Background task to run the fine-tuning process"""
    try:
        # # Load configuration
        # with hydra.initialize(config_path="../conf"):
        #     cfg = hydra.compose(config_name="config")

        logger.info(f"Starting fine-tuning for run_id: {run_id}")

        # # Get configurations from hydra config
        # model_config = cfg.model
        # lora_config = cfg.lora
        # dataset_config = cfg.dataset
        # training_config = cfg.training
        # output_config = cfg.output
        
        # Create a temporary directory for the entire process
        with tempfile.TemporaryDirectory() as temp_dir:
            ### DOWNLOAD DATASET FROM S3 FIRST ###
            logger.info(f"Loading and processing dataset from S3 for run_id: {run_id}")
            s3 = resources["s3_client"]
            AWS_S3_BUCKET = resources["s3_bucket"]
            
            # Define the object key (path in S3)
            object_key = f"{run_id}/data/train.jsonl"
            
            # Extract filename from object_key and create path in temp dir
            filename = os.path.basename(object_key)
            local_file_path = os.path.join(temp_dir, filename)
            
            try:
                # Download the file
                s3.download_file(AWS_S3_BUCKET, object_key, local_file_path)
                cfg_finetuning = s3.head_object(Bucket=AWS_S3_BUCKET, Key=object_key)
                logger.info(f"Successfully downloaded {object_key} to {local_file_path}")
                # Parse and convert the metadata to appropriate types
                raw_headers = cfg_finetuning["ResponseMetadata"]["HTTPHeaders"]
                cfg = parse_metadata(raw_headers)
                
                # Load the raw dataset
                raw_dataset = load_dataset(
                    "json", data_files=local_file_path, split=cfg["x-amz-meta-ft_dataset_split"]
                )
                
                # Standardize the format
                raw_dataset = standardize_sharegpt(raw_dataset)
                logger.info(f"Raw dataset loaded. Size: {len(raw_dataset)} samples")
                
            except Exception as e:
                logger.error(f"Error downloading or loading dataset: {e}")
                raise

            # Now that we've loaded the dataset, load the model and tokenizer
            logger.info(f"Loading model: {cfg["x-amz-meta-ft_model_name"]}")

            # Gemma Changes
            if "gemma" in cfg["x-amz-meta-ft_model_name"]:
                model, tokenizer = FastModel.from_pretrained(
                    model_name="unsloth/gemma-3-4b-it",
                    max_seq_length=2048,  # Choose any for long context!
                    load_in_4bit=True,  # 4 bit quantization to reduce memory
                    load_in_8bit=False,  # [NEW!] A bit more accurate, uses 2x memory
                    full_finetuning=False,  # [NEW!] We have full finetuning now!
                    # token = "hf_...", # use one if using gated models
                )
            else:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=cfg["x-amz-meta-ft_model_name"],
                    max_seq_length=cfg["x-amz-meta-ft_max_seq_length"],
                    dtype=None,
                    load_in_4bit=cfg["x-amz-meta-ft_load_in_4bit"],
                )
            logger.info(f"Model: {cfg["x-amz-meta-ft_model_name"]} loaded successfully")

            # Add LoRA adapters to the model
            logger.info("Adding LoRA adapters to the model")

            # Gemma Changes
            if "gemma" in cfg["x-amz-meta-ft_model_name"]:
                model = FastModel.get_peft_model(
                    model,
                    finetune_vision_layers=False,  # Turn off for just text!
                    finetune_language_layers=True,  # Should leave on!
                    finetune_attention_modules=True,  # Attention good for GRPO
                    finetune_mlp_modules=True,  # SHould leave on always!
                    r=cfg["x-amz-meta-ft_lora_r"],  # Larger = higher accuracy, but might overfit
                    lora_alpha=cfg["x-amz-meta-ft_lora_alpha"],  # Recommended alpha == r at least
                    lora_dropout=cfg["x-amz-meta-ft_lora_dropout"],
                    bias=cfg["x-amz-meta-ft_lora_bias"],
                    random_state=cfg["x-amz-meta-ft_random_state"],
                )
            else:
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=cfg["x-amz-meta-ft_lora_r"],
                    target_modules=cfg["x-amz-meta-ft_target_modules"],
                    lora_alpha=cfg["x-amz-meta-ft_lora_alpha"],
                    lora_dropout=cfg["x-amz-meta-ft_lora_dropout"],
                    bias=cfg["x-amz-meta-ft_lora_bias"],
                    use_gradient_checkpointing=cfg["x-amz-meta-ft_use_gradient_checkpointing"],
                    random_state=cfg["x-amz-meta-ft_random_state"],
                    use_rslora=cfg["x-amz-meta-ft_use_rslora"],
                    loftq_config=None,
                )
            logger.info("LoRA adapters added successfully")

            # Now process the dataset with the tokenizer
            logger.info("Processing dataset with tokenizer")
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=cfg["x-amz-meta-ft_chat_template"],
            )

            def formatting_prompts_func(examples):
                convos = examples["messages"]
                texts = [
                    tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    )
                    for convo in convos
                ]
                return {
                    "text": texts,
                }

            # Apply formatting to the dataset
            dataset = raw_dataset.map(
                formatting_prompts_func,
                batched=True,
            )
            logger.info(f"Dataset processed. Size: {len(dataset)} samples")

            logger.info("Initializing trainer")
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                dataset_text_field="text",
                max_seq_length=cfg["x-amz-meta-ft_max_seq_length"],
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                dataset_num_proc=1,
                packing=cfg["x-amz-meta-ft_packing"],
                args=TrainingArguments(
                    per_device_train_batch_size=cfg["x-amz-meta-ft_batch_size"],
                    gradient_accumulation_steps=cfg["x-amz-meta-ft_gradient_accumulation_steps"],
                    warmup_steps=cfg["x-amz-meta-ft_warmup_steps"],
                    num_train_epochs=1,  # Set this for 1 full training run.
                    # max_steps=cfg["x-amz-meta-ft_max_steps"],
                    learning_rate=cfg["x-amz-meta-ft_learning_rate"],
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=cfg["x-amz-meta-ft_weight_decay"],
                    lr_scheduler_type=cfg["x-amz-meta-ft_lr_scheduler_type"],
                    seed=cfg["x-amz-meta-ft_seed"],
                    output_dir=os.path.join(temp_dir, "outputs"),
                    report_to=None,
                ),
            )

            if "chatml" in cfg["x-amz-meta-ft_chat_template"]:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|im_start|>user\n",
                    response_part="<|im_start|>assistant\n",
                )

            # llama-3-8b has a different chat template then llama-3.2 / llama-3.1
            elif (
                "llama-3-8b" in cfg["x-amz-meta-ft_chat_template"]
                or "llama-3-8b" in cfg["x-amz-meta-ft_model_name"]
            ):
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|start_header_id|>user<|end_header_id|>",
                    response_part="<|start_header_id|>assistant<|end_header_id|>",
                )

            elif "llama" in cfg["x-amz-meta-ft_chat_template"] or "llama" in cfg["x-amz-meta-ft_model_name"]:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
                )

            elif "gemma" in cfg["x-amz-meta-ft_chat_template"] or "gemma" in cfg["x-amz-meta-ft_model_name"]:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<start_of_turn>user\n",
                    response_part="<start_of_turn>model\n",
                )
            else:
                logger.warning(
                    f"Chat template {cfg["x-amz-meta-ft_chat_template"]} not recognized. No training on responses only."
                )
                pass

            logger.info("Starting training...")

            # Show current memory stats
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.debug(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.debug(f"{start_gpu_memory} GB of memory reserved.")

            # Training the model
            trainer_stats = trainer.train()

            # Show final memory and time stats
            used_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            logger.info(
                f"{trainer_stats.metrics['train_runtime']} seconds used for training."
            )
            logger.info(
                f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
            )
            logger.debug(f"Peak reserved memory = {used_memory} GB.")
            logger.debug(
                f"Peak reserved memory for training = {used_memory_for_lora} GB."
            )
            logger.debug(f"Peak reserved memory % of max memory = {used_percentage} %.")
            logger.debug(
                f"Peak reserved memory for training % of max memory = {lora_percentage} %."
            )

            logger.info("Saving model artifacts to temporary directory...")

            # Create paths within the temp directory for model artifacts
            lora_model_temp_path = os.path.join(temp_dir, "lora_model")

            # Saving the final model as LoRA adapters to temp directory
            model.save_pretrained(lora_model_temp_path)
            tokenizer.save_pretrained(lora_model_temp_path)        

            logger.info("Model artifacts saved in temporary directory")

            # Upload model artifacts to S3
            logger.info("Uploading model artifacts to S3...")

            # Define S3 prefixes for the model artifacts
            lora_model_s3_prefix = f"{run_id}/lora_model"

            # Upload model directory using the shared S3 client
            lora_upload_success = upload_directory_to_s3(
                lora_model_temp_path, lora_model_s3_prefix, s3, AWS_S3_BUCKET
            )

            if lora_upload_success:
                logger.info(
                    f"Model artifacts for run {run_id} successfully uploaded to S3"
                )
            else:
                logger.warning(
                    f"Model artifacts for run {run_id} failed to upload to S3"
                )

            # Upload the models to huggingface hub
            hf_user  = os.getenv("HUGGINGFACE_USERNAME")
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

            # 2) Bail early with actionable warnings
            if not hf_user:
                logger.warning(
                    "HUGGINGFACE_USERNAME not set; skipping upload. "
                    "Please export HUGGINGFACE_USERNAME=<your-hf-username>."
                )
                return

            if not hf_token:
                logger.warning(
                    "HUGGINGFACE_TOKEN not found; skipping upload. "
                    "Please export HUGGINGFACE_TOKEN=<your-hf-token>."
                )
                return

            # 3) Build repo_id once
            model_name = cfg["x-amz-meta-ft_model_name"]
            repo_name  = f"{model_name}-{run_id}"
            repo_id    = f"{hf_user}/{repo_name}"

            # 4) Push both model + tokenizer in one try block
            try:
                logger.info(
                    f"Uploading model and tokenizer to Hugging Face Hub ({repo_id})"
                )
                model.push_to_hub(
                    repo_id=repo_id,
                    use_auth_token=hf_token,
                    private=True,
                )
                tokenizer.push_to_hub(
                    repo_id=repo_id,
                    use_auth_token=hf_token,
                    private=True,
                )
                logger.info(
                    f"Successfully uploaded run {run_id} artifacts to Hugging Face Hub ({repo_id})"
                )

            except Exception:
                logger.exception(
                    f"Failed to upload run {run_id} artifacts to Hugging Face Hub ({repo_id})"
                )

    except Exception as e:
        logger.error(f"Error in fine-tuning process for run {run_id}: {str(e)}")
        raise