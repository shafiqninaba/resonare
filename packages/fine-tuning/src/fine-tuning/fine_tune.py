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

    # Initialise configs
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    # Initialising the model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    logger.info("Model and tokenizer loaded successfully")

    logger.info("Adding LoRA adapters to the model")
    # Add LoRA adapters to the model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    logger.info("LoRA adapters added successfully")

    ### DATA PREP ###
    logger.info("Starting data preparation")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {
            "text": texts,
        }

    logger.info("Loading and processing dataset")
    dataset = load_dataset("mlabonne/FineTome-100k", split="train")
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
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
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
    model.save_pretrained("model_output/lora_model")  # Local saving
    tokenizer.save_pretrained("model_output/lora_model")

    # Save the model in 16bit merged format for vLLM
    model.save_pretrained_merged(
        "model_output/model",
        tokenizer,
        save_method="merged_16bit",
    )
    logger.info("Model artifacts saved successfully")


if __name__ == "__main__":
    main()
