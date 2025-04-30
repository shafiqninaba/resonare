from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import standardize_sharegpt
import hydra
from src.general_utils import setup_logger, setup_s3_client, upload_directory_to_s3
import os
import tempfile
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("fine_tuning_api")

# Dictionary to store connections and resources
resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup resources on startup
    logger.info("Setting up S3 client on application startup")
    resources["s3_client"] = setup_s3_client()
    resources["s3_bucket"] = os.getenv("AWS_S3_BUCKET")
    logger.info(f"S3 client initialized, using bucket: {resources['s3_bucket']}")

    yield

    # Clean up resources on shutdown
    logger.info("Cleaning up resources")
    resources.clear()
    logger.info("Resources cleaned up")


app = FastAPI(
    title="Model Fine-tuning API",
    description="API for fine-tuning language models",
    lifespan=lifespan,
)


class RunIDRequest(BaseModel):
    run_id: str


class TrainingResponse(BaseModel):
    status: str
    message: str
    run_id: str


def run_fine_tuning(run_id: str):
    """Background task to run the fine-tuning process"""
    try:
        # Load configuration
        with hydra.initialize(config_path="conf"):
            cfg = hydra.compose(config_name="config")

        logger.info(f"Starting fine-tuning for run_id: {run_id}")

        # Get configurations from hydra config
        model_config = cfg.model
        lora_config = cfg.lora
        dataset_config = cfg.dataset
        training_config = cfg.training
        output_config = cfg.output

        # Initialising the model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config.name,
            max_seq_length=model_config.max_seq_length,
            dtype=model_config.dtype,
            load_in_4bit=model_config.load_in_4bit,
        )
        logger.info(f"Model: {model_config.name} loaded successfully")

        logger.info("Adding LoRA adapters to the model")
        # Add LoRA adapters to the model
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
                dataset_num_proc=dataset_config.num_proc,
                packing=training_config.packing,
                args=TrainingArguments(
                    per_device_train_batch_size=training_config.per_device_train_batch_size,
                    gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                    warmup_steps=training_config.warmup_steps,
                    max_steps=training_config.max_steps,
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
            lora_model_s3_prefix = f"{run_id}/models/lora_model"

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


@app.post("/fine-tune", response_model=TrainingResponse)
async def fine_tune(request: RunIDRequest, background_tasks: BackgroundTasks):
    """
    Start a fine-tuning job with the specified run ID

    The run_id is used to identify the training data in S3 and to store the model artifacts.
    """
    run_id = request.run_id

    # Validate run_id
    if not run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id provided")

    # Validate S3 client exists
    if "s3_client" not in resources:
        raise HTTPException(status_code=500, detail="S3 client not initialized")

    # Add the fine-tuning task to background tasks
    background_tasks.add_task(run_fine_tuning, run_id)

    return TrainingResponse(
        status="started", message="Fine-tuning job started successfully", run_id=run_id
    )


@app.get("/health")
async def health_check():
    """Check if the API is running"""
    s3_status = "connected" if "s3_client" in resources else "disconnected"
    return {
        "status": "healthy",
        "message": "Fine-tuning API is operational",
        "s3_connection": s3_status,
    }


if __name__ == "__main__":
    try:
        with hydra.initialize(config_path="conf"):
            cfg = hydra.compose(config_name="config")
        uvicorn.run(
            "fastapi_main:app", host="0.0.0.0", port=8000, log_config=cfg.logging
        )
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        uvicorn.run("fastapi_main:app", host="0.0.0.0", port=8000)
