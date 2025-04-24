from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import standardize_sharegpt
import logging
from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging_config = OmegaConf.to_container(cfg.logging, resolve=True)

    # Initialize logging
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("fine_tuning")
    logger.info("Starting fine-tuning script.")

    # Get configurations from hydra config
    fine_tuning_config = cfg.fine_tuning
    model_config = fine_tuning_config.model
    lora_config = fine_tuning_config.lora
    dataset_config = fine_tuning_config.dataset
    training_config = fine_tuning_config.training
    output_config = fine_tuning_config.output

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

    logger.info(f"Loading and processing dataset from {dataset_config.path}")
    dataset = load_dataset(
        "json", data_files=dataset_config.path, split=dataset_config.split
    )
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
            num_train_epochs=1,
            # max_steps=training_config.max_steps,
            learning_rate=training_config.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=training_config.weight_decay,
            lr_scheduler_type=training_config.lr_scheduler_type,
            seed=training_config.seed,
            output_dir=output_config.dir,
            report_to=output_config.report_to,
        ),
    )

    logger.info("Starting training...")

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    logger.debug(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    logger.debug(f"{start_gpu_memory} GB of memory reserved.")

    # Training the model
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    logger.debug(f"Peak reserved memory = {used_memory} GB.")
    logger.debug(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    logger.debug(f"Peak reserved memory % of max memory = {used_percentage} %.")
    logger.debug(
        f"Peak reserved memory for training % of max memory = {lora_percentage} %."
    )

    logger.info("Saving model artifacts...")
    # Saving the final model as LoRA adapters
    model.save_pretrained(output_config.lora_model_path)
    tokenizer.save_pretrained(output_config.lora_model_path)

    # Save the model in 16bit merged format for vLLM
    model.save_pretrained_merged(
        output_config.merged_model_path,
        tokenizer,
        save_method=output_config.merged_save_method,
    )
    logger.info("Model artifacts saved successfully")


if __name__ == "__main__":
    main()
