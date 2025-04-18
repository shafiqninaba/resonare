# from unsloth import FastLanguageModel
# import torch
# from unsloth.chat_templates import get_chat_template
# from datasets import load_dataset
# from trl import SFTTrainer
# from transformers import TrainingArguments, DataCollatorForSeq2Seq
# from unsloth import is_bfloat16_supported
# from unsloth.chat_templates import standardize_sharegpt
# from unsloth.chat_templates import train_on_responses_only
import logging
from omegaconf import OmegaConf, DictConfig
import hydra


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging_config = OmegaConf.to_container(cfg.logging, resolve=True)

    # Initialize logging
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.info("Starting fine-tuning script.")


if __name__ == "__main__":
    main()
