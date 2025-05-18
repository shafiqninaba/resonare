from unsloth import FastLanguageModel, FastModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import standardize_sharegpt
import hydra
from src.general_utils import setup_logger, upload_directory_to_s3
import os
import tempfile
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

logger = setup_logger("fine_tuning")


def run_fine_tuning(run_id: str, resources: Dict[str, any]) -> None:
    """Background task to run the fine-tuning process"""
    try:
        # Load configuration
        with hydra.initialize(config_path="../conf"):
            cfg = hydra.compose(config_name="config")

        logger.info(f"Starting fine-tuning for run_id: {run_id}")

        # Get configurations from hydra config
        model_config = cfg.model
        lora_config = cfg.lora
        dataset_config = cfg.dataset
        training_config = cfg.training
        output_config = cfg.output

        # Initialising the model and tokenizer
        logger.info(f"Loading model: {model_config.name}")

        # Gemma Changes
        if "gemma" in model_config.name:
            model, tokenizer = FastModel.from_pretrained(
                model_name = "unsloth/gemma-3-4b-it",
                max_seq_length = 2048, # Choose any for long context!
                load_in_4bit = True,  # 4 bit quantization to reduce memory
                load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
                full_finetuning = False, # [NEW!] We have full finetuning now!
                # token = "hf_...", # use one if using gated models
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_config.name,
                max_seq_length=model_config.max_seq_length,
                dtype=model_config.dtype,
                load_in_4bit=model_config.load_in_4bit,
            )
        logger.info(f"Model: {model_config.name} loaded successfully")
 
        # Add LoRA adapters to the model
        logger.info("Adding LoRA adapters to the model")

        # Gemma Changes
        if "gemma" in model_config.name:
            model = FastModel.get_peft_model(
            model,
            finetune_vision_layers     = False, # Turn off for just text!
            finetune_language_layers   = True,  # Should leave on!
            finetune_attention_modules = True,  # Attention good for GRPO
            finetune_mlp_modules       = True,  # SHould leave on always!
            r = 8,           # Larger = higher accuracy, but might overfit
            lora_alpha = 8,  # Recommended alpha == r at least
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,)

        else:
            model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.r,
            target_modules=lora_config.target_modules,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
            random_state=lora_config.random_state,
            use_rslora=lora_config.use_rslora,
            loftq_config=lora_config.loftq_config,
        )

        logger.info("LoRA adapters added successfully")
            

        ### DATA PREP ###
        logger.info("Starting data preparation")
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=model_config.chat_template,
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

        logger.info(f"Loading and processing dataset from S3 for run_id: {run_id}")

        ### DOWNLOAD DATASET FROM S3 - USING SHARED CLIENT ###
        s3 = resources["s3_client"]
        AWS_S3_BUCKET = resources["s3_bucket"]

        # Define the object key (path in S3)
        object_key = f"{run_id}/data/train.jsonl"

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract filename from object_key and create path in temp dir
            filename = os.path.basename(object_key)
            local_file_path = os.path.join(temp_dir, filename)

            try:
                # Download the file
                s3.download_file(AWS_S3_BUCKET, object_key, local_file_path)
                logger.info(
                    f"Successfully downloaded {object_key} to {local_file_path}"
                )
                dataset = load_dataset(
                    "json", data_files=local_file_path, split=dataset_config.split
                )

            except Exception as e:
                logger.error(f"Error downloading file: {e}")
                raise

            dataset = standardize_sharegpt(dataset)
            dataset = dataset.map(
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
                max_seq_length=model_config.max_seq_length,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                dataset_num_proc=1,
                packing=training_config.packing,
                args=TrainingArguments(
                    per_device_train_batch_size=training_config.per_device_train_batch_size,
                    gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                    warmup_steps=training_config.warmup_steps,
                    num_train_epochs = 1, # Set this for 1 full training run.
                    # max_steps=training_config.max_steps,
                    learning_rate=training_config.learning_rate,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=training_config.weight_decay,
                    lr_scheduler_type=training_config.lr_scheduler_type,
                    seed=training_config.seed,
                    output_dir=os.path.join(temp_dir, output_config.dir),
                    report_to=output_config.report_to,
                ),
            )
            
            if "chatml" in model_config.chat_template:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part = "<|im_start|>user\n",
                    response_part = "<|im_start|>assistant\n",
                )

            # llama-3-8b has a different chat template then llama-3.2 / llama-3.1
            elif "llama-3-8b" in model_config.chat_template or "llama-3-8b" in model_config.name:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part = "<|start_header_id|>user<|end_header_id|>",
                    response_part = "<|start_header_id|>assistant<|end_header_id|>",
                )

            elif "llama" in model_config.chat_template or "llama" in model_config.name:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
                )

            elif "gemma" in model_config.chat_template or "gemma" in model_config.name:
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part = "<start_of_turn>user\n",
                    response_part = "<start_of_turn>model\n",
                )
            else:
                logger.warning(
                    f"Chat template {model_config.chat_template} not recognized. No training on responses only."
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
                f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
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

    except Exception as e:
        logger.error(f"Error in fine-tuning process for run {run_id}: {str(e)}")
        raise
