from unsloth import FastLanguageModel
from transformers import TextStreamer
from src.general_utils import setup_logger, downloadDirectoryFroms3
import tempfile
from dotenv import load_dotenv
import os


def main(run_id):
    """Main function to run the inference"""
    load_dotenv()

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    # Set up logger
    logger = setup_logger("inference")

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Created temporary directory: {temp_dir}")

        # Download the model from S3 to the temporary directory
        dir_name = f"{run_id}/models/lora_model"
        downloadDirectoryFroms3(os.getenv("AWS_S3_BUCKET"), dir_name, temp_dir)

        # Load the model and tokenizer from the temporary directory
        logger.info("Loading model and tokenizer...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=temp_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

        messages = [
            {"role": "user", "content": ">>> bro what's your name ah?"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            input_ids=inputs,
            streamer=text_streamer,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )


if __name__ == "__main__":
    main("b28c5ef19549486c9f09544ea1428162")
