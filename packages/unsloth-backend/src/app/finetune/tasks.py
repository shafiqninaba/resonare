import os
import tempfile

import boto3
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import (
    get_chat_template,
    standardize_sharegpt,
    train_on_responses_only,
)

from ..core.s3_client import upload_directory_to_s3
from .utils import parse_metadata, setup_logger

load_dotenv()

logger = setup_logger("fine_tuning")


def run_fine_tuning(run_id: str, s3_client: boto3.client, bucket_name: str) -> None:
    """Background task to run the fine-tuning process end-to-end.

    This function orchestrates downloading the dataset from S3, loading
    and adapting the pretrained model, training with LoRA adapters, and
    uploading the resulting adapters back to S3 (and optionally to Hugging Face Hub).

    Args:
        run_id (str):
            Unique identifier for this fine-tuning run.  
            Used to locate data and to namespace output artifacts.
        s3_client (boto3.client):
            Authenticated boto3 S3 client for downloading/uploading files.
        bucket_name (str):
            Name of the S3 bucket where training data and model artifacts reside.

    Raises:
        RuntimeError:
            If any step in the process fails (dataset download, model load,
            training, or upload), an error is logged and re-raised.

    Workflow Summary:
        1. **Dataset download & metadata**  
           • Construct S3 key: `{run_id}/data/train.jsonl`  
           • Download to a temporary directory.  
           • Read S3 object metadata (HTTPHeaders) for training config.

        2. **Model & tokenizer loading**  
           • Based on `ft_model_name` metadata, choose between `FastLanguageModel`  
             or `FastModel`.  
           • Load with quantization flags (`4bit`, `8bit`, etc.).

        3. **LoRA adapter setup**  
           • Attach lightweight LoRA adapters to the base model.  
           • Configure adapter hyperparameters (r, alpha, dropout, etc.)  
             from metadata.

        4. **Data preparation**  
           • Standardize raw JSON into conversation blocks.  
           • Apply the chosen chat template to produce text inputs.

        5. **Trainer initialization**  
           • Instantiate `SFTTrainer` with model, tokenizer, dataset,  
             data collator, and `TrainingArguments` (batch size, LR, epochs).

        6. **Fine-tuning**  
           • Optionally wrap trainer via `train_on_responses_only` based on template.  
           • Execute `trainer.train()` and log GPU memory / time statistics.

        7. **Artifact saving & upload**  
           • Save LoRA adapters (model & tokenizer) locally under `lora_model/`.  
           • Upload the folder back to S3 at `{run_id}/lora_model`.  
           • If `HUGGINGFACE_USERNAME` & `HUGGINGFACE_TOKEN` are set,  
             push to your private HF repo.

        8. **Error handling**  
           • Any exception is logged in detail and then re-raises to mark the job as failed.
    """
    try:
        logger.info(f"Starting fine-tuning for run_id: {run_id}")

        # Now, we load the configuration from the metadata in train.jsonl rather than hydra config
        # # Load configuration
        # with hydra.initialize(config_path="../conf"):
        #     cfg = hydra.compose(config_name="config")

        # # Get configurations from hydra config
        # model_config = cfg.model
        # lora_config = cfg.lora
        # dataset_config = cfg.dataset
        # training_config = cfg.training
        # output_config = cfg.output

        # Create a temporary directory for the entire process
        with tempfile.TemporaryDirectory() as temp_dir:
            # --------------------------------------------------------------------
            # Download the dataset from S3 and getting training parameters from metadata
            # --------------------------------------------------------------------
            logger.info(f"Loading and processing dataset from S3 for run_id: {run_id}")
            AWS_S3_BUCKET = bucket_name

            # Define the object key (path in S3)
            object_key = f"{run_id}/data/train.jsonl"

            # Extract filename from object_key and create path in temp dir
            filename = os.path.basename(object_key)
            local_file_path = os.path.join(temp_dir, filename)

            try:
                # Download the file
                s3_client.download_file(AWS_S3_BUCKET, object_key, local_file_path)
                cfg_finetuning = s3_client.head_object(
                    Bucket=AWS_S3_BUCKET, Key=object_key
                )
                logger.info(
                    f"Successfully downloaded {object_key} to {local_file_path}"
                )
                # Parse and convert the metadata to appropriate types
                raw_headers = cfg_finetuning["ResponseMetadata"]["HTTPHeaders"]
                cfg = parse_metadata(raw_headers)

                # Load the raw dataset
                raw_dataset = load_dataset(
                    "json",
                    data_files=local_file_path,
                    split=cfg["x-amz-meta-ft_dataset_split"],
                )

                # Standardize the format
                raw_dataset = standardize_sharegpt(raw_dataset)
                logger.info(f"Raw dataset loaded. Size: {len(raw_dataset)} samples")

            except Exception as e:
                logger.error(f"Error downloading or loading dataset: {e}")
                raise

            # --------------------------------------------------------------------
            # Loading the model
            # --------------------------------------------------------------------
            # Now that we've loaded the dataset, load the model and tokenizer
            logger.info(f"Loading model: {cfg['x-amz-meta-ft_model_name']}")

            # Gemma 3 is a vision model, so we need to use FastModel to load it
            if "gemma-3" in cfg["x-amz-meta-ft_model_name"]:
                model, tokenizer = FastModel.from_pretrained(
                    model_name=cfg["x-amz-meta-ft_model_name"],
                    max_seq_length=cfg[
                        "x-amz-meta-ft_max_seq_length"
                    ],  # Choose any for long context!
                    load_in_4bit=cfg[
                        "x-amz-meta-ft_load_in_4bit"
                    ],  # 4 bit quantization to reduce memory
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
            logger.info(f"Model: {cfg['x-amz-meta-ft_model_name']} loaded successfully")

            # --------------------------------------------------------------------
            # Add LoRA adapters to the model
            # --------------------------------------------------------------------
            logger.info("Adding LoRA adapters to the model")

            # Gemma 3 is a vision model, so we need to use FastModel to load it
            if "gemma-3" in cfg["x-amz-meta-ft_model_name"]:
                model = FastModel.get_peft_model(
                    model,
                    finetune_vision_layers=False,  # Turn off for just text!
                    finetune_language_layers=True,  # Should leave on!
                    finetune_attention_modules=True,  # Attention good for GRPO
                    finetune_mlp_modules=True,  # SHould leave on always!
                    r=cfg[
                        "x-amz-meta-ft_lora_r"
                    ],  # Larger = higher accuracy, but might overfit
                    lora_alpha=cfg[
                        "x-amz-meta-ft_lora_alpha"
                    ],  # Recommended alpha == r at least
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
                    use_gradient_checkpointing=cfg[
                        "x-amz-meta-ft_use_gradient_checkpointing"
                    ],
                    random_state=cfg["x-amz-meta-ft_random_state"],
                    use_rslora=cfg["x-amz-meta-ft_use_rslora"],
                    loftq_config=None,
                )
            logger.info("LoRA adapters added successfully")

            # --------------------------------------------------------------------
            # Tokenize the dataset and prepare it for training
            # --------------------------------------------------------------------
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

            # --------------------------------------------------------------------
            # Initialize the trainer
            # --------------------------------------------------------------------
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
                    gradient_accumulation_steps=cfg[
                        "x-amz-meta-ft_gradient_accumulation_steps"
                    ],
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

            # --------------------------------------------------------------------
            # Configure trainer for training on responses only based on the chat template
            # --------------------------------------------------------------------
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

            elif (
                "llama" in cfg["x-amz-meta-ft_chat_template"]
                or "llama" in cfg["x-amz-meta-ft_model_name"]
            ):
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
                )

            elif (
                "gemma" in cfg["x-amz-meta-ft_chat_template"]
                or "gemma" in cfg["x-amz-meta-ft_model_name"]
            ):
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part="<start_of_turn>user\n",
                    response_part="<start_of_turn>model\n",
                )
            else:
                logger.warning(
                    f"Chat template {cfg['x-amz-meta-ft_chat_template']} not recognized. Defaulting to standard training (no train on responses only applied)."
                )
                pass

            # --------------------------------------------------------------------
            # Show current memory stats
            # --------------------------------------------------------------------
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.debug(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.debug(f"{start_gpu_memory} GB of memory reserved.")

            # --------------------------------------------------------------------
            # TRAINING
            # --------------------------------------------------------------------
            logger.info("Starting training...")
            trainer_stats = trainer.train()

            # --------------------------------------------------------------------
            # Show final memory and time stats
            # --------------------------------------------------------------------
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

            # --------------------------------------------------------------------
            # Try to upload the trained models to S3
            # --------------------------------------------------------------------
            logger.info("Saving model artifacts to temporary directory...")
            try:
                lora_model_temp_path = os.path.join(
                    temp_dir, "lora_model"
                )  # Create paths within the temp directory for model artifacts

                # Saving the final model as LoRA adapters to temp directory
                model.save_pretrained(lora_model_temp_path)
                tokenizer.save_pretrained(lora_model_temp_path)

                logger.info("Model artifacts sucessfully saved in temporary directory!")

            except Exception as e:
                logger.error(f"Error saving model artifacts in temp directory: {e}")
                raise

            # Upload model artifacts to S3
            logger.info("Uploading model artifacts to S3...")

            lora_model_s3_prefix = (
                f"{run_id}/lora_model"  # Define S3 prefixes for the model artifacts
            )

            # Upload model directory using the shared S3 client
            lora_upload_success = upload_directory_to_s3(
                lora_model_temp_path, lora_model_s3_prefix, s3_client, AWS_S3_BUCKET
            )
            if lora_upload_success:
                logger.info(
                    f"Model artifacts for run {run_id} successfully uploaded to S3"
                )
            else:
                logger.warning(
                    f"Model artifacts for run {run_id} failed to upload to S3"
                )

            # --------------------------------------------------------------------
            # Try to upload the trained models to huggingface hub
            # --------------------------------------------------------------------
            hf_user = os.getenv("HUGGINGFACE_USERNAME")
            hf_token = os.getenv("HUGGINGFACE_TOKEN")

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

            model_name = cfg["x-amz-meta-ft_model_name"].split("/")[-1]
            repo_name = f"{model_name}-{run_id}"
            repo_id = f"{hf_user}/{repo_name}"

            try:
                logger.info(
                    f"Uploading model and tokenizer to Hugging Face Hub: ({repo_id})"
                )
                model.push_to_hub(
                    repo_id=repo_id,
                    token=hf_token,
                    private=True,
                )
                tokenizer.push_to_hub(
                    repo_id=repo_id,
                    token=hf_token,
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
