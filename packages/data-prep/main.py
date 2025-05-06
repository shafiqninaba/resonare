#!/usr/bin/env python3
"""
Processes a JSON chat export (e.g., from Telegram) into filtered conversation
sessions, chunked by time and token limits, suitable for further analysis or
LLM fine-tuning.

This script performs the following steps:
0. Sets up logging configuration using either standard logging configuration file or Loguru (more pythonic).

1. Loads a model specific tokenizer using HuggingFace Transformers (fallback to TikToken).

2. Setup S3 client for uploading processed data if needed.

3. Loads the raw JSON export from a single file or a directory of files.
    - If a file is specified, it processes that single JSON file, which should contain a list of chats.
    - If a directory is specified, it processes all JSON files within it, which should each contain a single chat.

4. From the loaded raw data, start building `Chat` objects containing `Message` objects
   filtering for textual content and applying the date limit if specified.

5. Chunks messages within each chat into conversation 'blocks' based on time gaps
   (`convo_block_thereshold_secs`) and token counts (`min_tokens_per_block`,
   `max_tokens_per_block`). Discards blocks outside the token range. Discards conversations with no blocks.

6. Merges consecutive messages from the same sender within each block,
   prefixing each original message line with a delimiter (`message_delimiter`).
   Optionally prepends a system message to each block if specified in config.

7. Calculates and logs summary statistics about the processed chats and blocks.

8. Exports the processed data into two JSON / JSONL files:
   - One file containing full chat metadata and all associated blocks.
   - One file containing only the processed blocks, one block per line,
     suitable for ML training pipelines.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import boto3
import hydra
from omegaconf import DictConfig

from src.models import Chat, Message
from src.utils.general import setup_standard_logging
from src.utils.processing import (
    calculate_chat_stats,
    load_tokenizer,
    parse_date_limit,
)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---------------------
    # 0) Set up logging configuration
    # ---------------------
    # Get absolute path to the directory of main.py
    # We assume config folder containing logging conf is always in the same directory as main.py
    # Avoid issues when running main.py from different directories
    project_root = Path(__file__).parent

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")

    setup_standard_logging(
        logging_config_path=os.path.join(
            project_root,
            "conf",
            "logging.yaml",
        ),
    )

    # setup_loguru_logging(logs_dir=os.path.join(project_root, "logs"))

    # ---------------------
    # 1) Tokenizer loading
    # ---------------------
    # We need a tokenizer to split chat messages into tokens and obtain token counts, for chunking and filtering:
    #  - Prefer the same tokenizer family (e.g. BPE, SentencePiece, WordPiece) as our target finetuning model for accuracy.
    #  - Use HuggingFace’s AutoTokenizer to load the specific tokenizer for the target model.
    #  - If that fails, fall back to OpenAI’s tiktoken (BPE) for speed and API‑compatibility.
    logger.info(f"Loading tokenizer for model {cfg.model_id} for token counting...")
    tokenizer = load_tokenizer(model_name=cfg.model_id)

    # ---------------------
    # 2) Setup S3 client for uploading processed data if needed
    # ---------------------
    s3 = None
    if "s3" in cfg.output.modes:
        logger.info("Setting up S3 client for uploading processed data...")

        try:
            # # Read AWS credentials from environment variables
            # Note: boto3 looks for AWS credentials in serveal locations. https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

            # AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
            # AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

            # if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
            #     raise EnvironmentError(
            #         "Missing AWS credentials in environment variables."
            #     )

            # Create S3 client
            s3 = boto3.client(
                "s3",
                # aws_access_key_id=AWS_ACCESS_KEY,
                # aws_secret_access_key=AWS_SECRET_KEY,
                region_name=cfg.output.s3_region,
            )
            logger.info("S3 client setup successful.")

        except Exception as e:
            logger.error(
                f"Failed to set up S3 client: {e}, s3 upload will be skipped, only local export will be used."
            )
            cfg.output.modes = ["local"]

    # --------------------------------------------
    # 3.1) Load and upload raw data from JSON data
    # --------------------------------------------
    # We load the raw JSON export from a single file or a directory of files.
    # if a file is specified, it processes that single JSON file, which should contain a list of individual chats.
    # result.json: {chats: {list: {adam: {messages: []}, "zack": {messages: []}}}}
    # if a directory is specified, it processes all JSON files within it, which should each contain a single chat.
    # result.json: {name: "adam", type: "personal_chat", messages: [{from: "adam", text_entities: [], sticker_emoji: ""}]
    logger.info("Loading chats from raw JSON data...")

    mode = cfg.input.mode.lower()
    if mode not in ["file", "dir"]:
        logger.error(
            f"Invalid raw_input.mode: {mode}; must be 'file' or 'dir', defaulting to 'file'"
        )
        mode = "file"

    # Handle a single export file (e.g., "result.json")
    if mode == "file":
        fp = Path(cfg.input.file)
        if not fp.is_file():
            logger.error(f"Raw JSON export file not found: {fp}")
            raise FileNotFoundError(fp)

        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            raw_chats = data["chats"]["list"]  # Extract the list of chats
        except Exception as e:
            logger.error(f"Failed to load raw JSON export file {fp}: {e}")
            raise

    # Handle a directory of individual chat JSON files
    elif mode == "dir":
        dd = Path(cfg.input.dir)
        if not dd.is_dir():
            logger.error(f"Raw JSON directory not found: {dd}")
            raise FileNotFoundError(dd)

        raw_chats = []
        for chat_file in sorted(dd.glob("*.json")):
            try:
                chat_data = json.loads(chat_file.read_text(encoding="utf-8"))
                raw_chats.append(chat_data)
            except Exception as e:
                logger.error(f"Failed to load raw JSON export file {chat_file}: {e}")
                continue

    logger.info(f"Loaded {len(raw_chats)} chats for processing")

    # -------------------------------
    # 3.2) Export: Raw Chats
    # -------------------------------
    # Since we have no user feature for now, we treat each run as a separate run.
    # We create a new directory for each run, and store the processed data there.
    # UUID is used to create a unique directory name for each run.

    run_id = uuid.uuid4().hex  # Generate a fresh run UUID with no dashes

    # Define base paths
    base_dir = Path(cfg.output.local_dir)
    run_dir = base_dir / run_id

    # Save locally if configured
    if "local" in cfg.output.modes:
        logger.info(f"Saving raw chats locally to {run_dir}...")

        run_dir.mkdir(parents=True, exist_ok=True)
        raw_chats_filepath = run_dir / "raw.json"

        with raw_chats_filepath.open("w", encoding="utf-8") as f:
            json.dump(raw_chats, f, ensure_ascii=False, indent=2)

    # Upload to S3 if configured
    if "s3" in cfg.output.modes:
        logger.info(f"Uploading raw chats to S3 bucket {cfg.output.s3_bucket}...")

        try:
            s3.put_object(
                Bucket=cfg.output.s3_bucket,
                Key=f"{run_id}/data/raw.json",
                Body=json.dumps(raw_chats, ensure_ascii=False, indent=2),
                Metadata={
                    "uuid": run_id,
                },
            )
            logger.info(
                f"Successfully uploaded raw chats to s3://{cfg.output.s3_bucket}/{run_id}/raw.json"
            )
        except Exception as e:
            logger.error(f"Failed to upload raw chats to S3: {e}")
            raise

    # --------------------------------------------
    # 4) Build Chat objects
    # --------------------------------------------
    # We assemble a list of Chat instances, each representing a chat:
    #  - contact_name: the person or group name

    #  - chat_type: one‑on‑one, group, or supergroup

    #  - messages: a flat, chronological list of Message objects, each with:
    #    - role: the sender of the message (user or system)
    #    - content: the message text
    #    - timestamp: the datetime when the message was sent

    #  - blocks: A list of message blocks. Each block is a list of temporally
    #            and contextually related messages, chunked according to time
    #            and token limits. Defaults to an empty list.
    logger.info("Building chat objects from raw chats...")

    chats: List[Chat] = []
    target_name = cfg.target_name  # Name identifying "our" side of the conversation, renamed to "assistant" in the output
    date_limit = parse_date_limit(
        cfg.date_limit  # Optional date limit for filtering messages
    )

    for chat in raw_chats:
        contact_name = chat.get("name")

        if not contact_name:  # Skip chats without a name (deleted/anonymous)
            continue

        chat_type = chat.get("type")
        # Currently limiting to personal chats.
        # TODO: Expand this list or logic if group chat support is added @renhwa.
        # Potential Issue: Group chats seem to be a little wonky, only includes target name and messages.
        if chat_type not in ["personal_chat"]:
            continue

        msgs: List[Message] = []
        for msg in chat.get("messages", []):
            try:
                sender = msg.get("from", "")
                ents = msg.get("text_entities", [])
                sticker = msg.get("sticker_emoji", "")

                # We need a sender and some form of text content (entities or sticker).
                # We only include text_entities and sticker_emoji, since those produce tokenizable text, and skip other media (photos, files, voice notes
                if not sender or (not ents and not sticker):
                    continue

                # Reconstruct the textual content from entities + emoji.
                raw_text = "".join(ent["text"] for ent in ents) + sticker
                # Remove leading/trailing whitespace and replace internal newlines with spaces.
                # Newlines will be used later to delimit merged messages.
                content = raw_text.strip().replace("\n", " ")

                # Parse timestamp and apply date filter if set.
                timestamp = datetime.fromisoformat(msg["date"])
                if date_limit and timestamp < date_limit:
                    continue

                msgs.append(
                    Message(
                        role="assistant"  # Assign role based on sender
                        if sender == target_name
                        else "user",  # Assign sender role based on target_name
                        content=content,
                        timestamp=timestamp,
                    )
                )

            except Exception as e:
                logger.warning(
                    f"[{contact_name}] skipping a message due to parse error: {e}"
                )

        # If we found any valid messages, add the Conversation object.
        if msgs:
            msgs.sort(
                key=lambda m: m.timestamp
            )  # Sort messages by timestamp to ensure chronological order.

            chats.append(
                Chat(
                    contact_name=contact_name,
                    type=chat_type,
                    messages=msgs,
                )
            )

    logger.info(f"Built {len(chats)} usable chat objects.")

    # -------------------------------
    # 5) Chunking each chat into conversation 'blocks'
    # -------------------------------
    # Split each Chat.messages into “blocks” so that each block:
    #   • Maintains temporal context (messages no more than time_threshold_sec apart)
    #   • Stays within a token-budget (min_tokens ≤ block_tokens ≤ max_tokens)
    # This ensures that during LLM training each example has coherent context, and is neither too short (unhelpful) nor too long (slow to train on).

    convo_thereshold_secs = cfg.convo_block_thereshold_secs
    min_tokens = cfg.min_tokens_per_block
    max_tokens = cfg.max_tokens_per_block

    # Sanity‑checks
    if min_tokens >= max_tokens:
        logger.warning(
            f"Invalid token thresholds: min_tokens ({min_tokens}) ≥ max_tokens ({max_tokens}). "
            "Resetting to defaults: min_tokens=100, max_tokens=3000."
        )
        min_tokens, max_tokens = 100, 3000

    # variables to track block counts
    num_short_blocks = 0
    num_long_blocks = 0

    logger.info("Chunking chats into blocks...")
    for chat in chats:
        chat.blocks = []
        current_block: List[Message] = []
        current_tokens = 0
        previous_time: Optional[datetime] = None

        for msg in chat.messages:
            gap = (
                (msg.timestamp - previous_time).total_seconds()
                if previous_time
                else None
            )
            msg_tokens = len(tokenizer.encode(msg.content))

            # Continue block if within time and token limits
            if (
                previous_time
                and gap <= convo_thereshold_secs
                and (current_tokens + msg_tokens) <= max_tokens
            ):
                current_block.append(msg)
                current_tokens += msg_tokens
            else:
                # Commit the existing block
                if current_block:
                    if min_tokens <= current_tokens <= max_tokens:
                        chat.blocks.append(current_block)
                    elif current_tokens < min_tokens:
                        num_short_blocks += 1
                    else:
                        num_long_blocks += 1

                # Start a new block
                current_block = [msg]
                current_tokens = msg_tokens

            previous_time = msg.timestamp

        # Commit any remaining block
        if current_block:
            if min_tokens <= current_tokens <= max_tokens:
                chat.blocks.append(current_block)
            elif current_tokens < min_tokens:
                num_short_blocks += 1
            else:
                num_long_blocks += 1

    # Discard empty chats with empty blocks
    num_original_chats = len(chats)
    chats = [c for c in chats if c.blocks]
    num_discarded_chats = num_original_chats - len(chats)

    # Log the results
    num_total_blocks = sum(len(c.blocks) for c in chats)
    logger.info(
        f"Chunking complete: {num_original_chats} conversations → {len(chats)} conversations ({num_discarded_chats} discarded due to empty blocks), "
        f"{num_total_blocks} chat blocks created; {num_short_blocks} too short, {num_long_blocks} too long."
    )

    # -------------------------------
    # 6) Merge consecutive messages by sender within each block
    # -------------------------------
    # For each block in each Conversation, we:
    #   • Group consecutive messages from the same sender into one Message
    #   • Prefix every line with the delimiter (e.g. '>>>')
    #   • Separate lines with '\n'
    #   • Keep the timestamp of the first message in each group
    #   • For each block, trim leading assistant messages and trailing user messages
    #   • Add a system message at the start of each block if specified
    logger.info("Merging consecutive messages by sender within each block...")
    delimiter = cfg.message_delimiter.strip()

    for convo in chats:
        merged_blocks: List[List[Message]] = []

        for block in convo.blocks:
            merged_messages: List[Message] = []

            first_msg = block[0]
            current_sender = first_msg.role
            current_timestamp = first_msg.timestamp
            current_content = f"{delimiter} {first_msg.content.strip()}"

            for msg in block[1:]:
                if (
                    msg.role == current_sender
                ):  # concatenate messages from the same sender
                    current_content += f"\n{delimiter} {msg.content.strip()}"
                else:
                    # Create and add the merged message to the list
                    merged_messages.append(
                        Message(
                            role=current_sender,
                            content=current_content,
                            timestamp=current_timestamp,
                        )
                    )
                    current_sender = msg.role
                    current_timestamp = msg.timestamp
                    current_content = f"{delimiter} {msg.content.strip()}"

            # Add last merged message if exists
            if current_content:
                merged_messages.append(
                    Message(
                        role=current_sender,
                        content=current_content,
                        timestamp=current_timestamp,
                    )
                )

            merged_blocks.append(merged_messages)

        convo.blocks = merged_blocks

    # ------------------------------------------------------------------
    # 6b) Ensure each block starts with USER and ends with ASSISTANT
    # ------------------------------------------------------------------
    fixed_blocks_total = 0
    discarded_blocks = 0

    for convo in chats:
        new_blocks: List[List[Message]] = []

        for block in convo.blocks:
            if not block:
                continue

            # Trim leading assistant messages
            while block and block[0].role == "assistant":
                block = block[1:]

            # Trim trailing user messages
            while block and block[-1].role == "user":
                block = block[:-1]

            # Keep block only if it now starts with user and ends with assistant
            if (
                len(block) >= 2
                and block[0].role == "user"
                and block[-1].role == "assistant"
            ):
                new_blocks.append(block)
                fixed_blocks_total += 1
            else:
                discarded_blocks += 1

        convo.blocks = new_blocks

    logger.info(
        f"Role‑sanity pass complete: {fixed_blocks_total} blocks kept, "
        f"{discarded_blocks} blocks discarded for incorrect start/end roles."
    )

    # Append the system message to each block
    if cfg.system_prompt:
        logger.info(
            f"Prepending system message to each conversation block with content: {cfg.system_prompt}"
        )
        # Try to build the system message first
        try:
            system_message = Message(
                role="system",
                content=cfg.system_prompt,
                timestamp=None,
            )
        except Exception as e:
            logger.error(
                f"Failed to create system message, skipping system prompts: {e}"
            )
            system_message = None

        # 2) If creation succeeded, prepend to every block
        if system_message:
            for convo in chats:
                for block in convo.blocks:
                    try:
                        block.insert(0, system_message)
                    except Exception as e:
                        # log and move on to the next block
                        logger.warning(
                            f"Could not prepend system message for chat '{convo.contact_name}', "
                            f"block starting at {block[0].timestamp if block else 'unknown'}: {e}"
                        )
                        continue

    # -------------------------------
    # 7) Log summary statistics
    # -------------------------------
    logger.info("Calculating statistics of processed chats...")
    chat_stats = calculate_chat_stats(chats, tokenizer)

    # Define the number of top entries to display
    k = 10

    # Extract and sort the block breakdown by the number of blocks in descending order
    top_k_breakdown = sorted(
        chat_stats["block_breakdown"].items(), key=lambda item: item[1], reverse=True
    )[:k]

    stats_table = "\n"
    stats_table += "*" * 36 + "\n"
    stats_table += "*{:^34}*\n".format("Chat Statistics Summary")
    stats_table += "*" * 36 + "\n"
    stats_table += f"{'Metric':<25} | {'Value':>8}\n"
    stats_table += "-" * 36 + "\n"
    stats_table += f"{'Total Chats':<25} | {chat_stats['num_chats']:>8}\n"
    stats_table += f"{'Total Blocks':<25} | {chat_stats['num_blocks']:>8}\n"
    stats_table += (
        f"{'Min Tokens/Block':<25} | {chat_stats['min_tokens_per_block']:>8}\n"
    )
    stats_table += (
        f"{'Max Tokens/Block':<25} | {chat_stats['max_tokens_per_block']:>8}\n"
    )
    stats_table += (
        f"{'Avg Tokens/Block':<25} | {chat_stats['avg_tokens_per_block']:>8.2f}\n"
    )
    stats_table += f"{'Min Duration (min)':<25} | {chat_stats['min_duration_minutes_per_block']:>8.2f}\n"
    stats_table += f"{'Max Duration (min)':<25} | {chat_stats['max_duration_minutes_per_block']:>8.2f}\n"
    stats_table += f"{'Avg Duration (min)':<25} | {chat_stats['avg_duration_minutes_per_block']:>8.2f}\n"

    stats_table += "\n"
    stats_table += "*" * 36 + "\n"
    stats_table += "*{:^34}*\n".format("Top Chats by Block Count")
    stats_table += "*" * 36 + "\n"
    for rank, (name, count) in enumerate(top_k_breakdown, start=1):
        stats_table += f"{rank:>2}. {name:<28} {count:>5}\n"

    logger.info("\n" + stats_table)

    # -------------------------------
    # 8) Export: Processed Chats and Training Blocks
    # -------------------------------
    logger.info("Exporting processed chats and training blocks...")

    # Define paths
    processed_chats_filepath = run_dir / "processed.json"
    training_blocks_filepath = run_dir / "train.jsonl"

    # --- Manually Define Metadata ---
    metadata_dict = {
        "uuid": run_id,
        "model_id": cfg.model_id,
        "target_name": cfg.target_name,
        "system_prompt": cfg.system_prompt,
        "date_limit": str(cfg.date_limit) if cfg.date_limit else "None",
        "convo_block_thereshold_secs": str(cfg.convo_block_thereshold_secs),
        "min_tokens_per_block": str(cfg.min_tokens_per_block),
        "max_tokens_per_block": str(cfg.max_tokens_per_block),
        "message_delimiter": cfg.message_delimiter,
    }
    metadata_dict.update(
        {f"stats_{k}": str(v) for k, v in chat_stats.items()}
    )  # add stats to metadata

    # 8.1) Prepare chat records
    logger.info("Preparing processed chat records...")
    chat_records = []
    for chat in chats:
        chat_record = {
            "contact_name": chat.contact_name,
            "chat_type": chat.type,
            "num_blocks": len(chat.blocks),
            "blocks": [
                {
                    "messages": [
                        {
                            "timestamp": msg.timestamp.isoformat()
                            if msg.timestamp
                            else None,
                            "role": msg.role,
                            "content": msg.content,
                        }
                        for msg in block
                    ]
                }
                for block in chat.blocks
            ],
        }
        chat_records.append(chat_record)

    # 8.2) Save processed chats locally if needed
    if "local" in cfg.output.modes:
        logger.info(f"Saving processed chats locally to {processed_chats_filepath}...")
        with processed_chats_filepath.open("w", encoding="utf-8") as f:
            json.dump(chat_records, f, ensure_ascii=False, indent=2)

    # 8.3) Upload processed chats to S3 if needed
    if "s3" in cfg.output.modes and s3 is not None:
        logger.info(f"Uploading processed chats to S3 bucket {cfg.output.s3_bucket}...")
        try:
            s3.put_object(
                Bucket=cfg.output.s3_bucket,
                Key=f"{run_id}/data/processed.json",
                Body=json.dumps(chat_records, ensure_ascii=False, indent=2),
                Metadata=metadata_dict,
            )
            logger.info(
                f"Successfully uploaded chats.json to s3://{cfg.output.s3_bucket}/{run_id}/data/processed.json"
            )
        except Exception as e:
            logger.error(f"Failed to upload chats.json to S3: {e}")

    # 8.4) Save training blocks locally if needed
    logger.info("Preparing training blocks...")
    training_block_lines = []
    for chat in chats:
        for block in chat.blocks:
            record = {
                "messages": [
                    {"role": msg.role, "content": msg.content} for msg in block
                ]
            }
            training_block_lines.append(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )

    if "local" in cfg.output.modes:
        logger.info(f"Saving training blocks locally to {training_blocks_filepath}...")
        with training_blocks_filepath.open("w", encoding="utf-8") as f:
            f.write("\n".join(training_block_lines))

    # 8.5) Upload training blocks to S3 if needed
    if "s3" in cfg.output.modes and s3 is not None:
        logger.info(f"Uploading training blocks to S3 bucket {cfg.output.s3_bucket}...")
        try:
            s3.put_object(
                Bucket=cfg.output.s3_bucket,
                Key=f"{run_id}/data/train.jsonl",
                Body="\n".join(training_block_lines),
                Metadata=metadata_dict,
            )
            logger.info(
                f"Successfully uploaded train.jsonl to s3://{cfg.output.s3_bucket}/{run_id}/data/train.jsonl"
            )
        except Exception as e:
            logger.error(f"Failed to upload train.jsonl to S3: {e}")


if __name__ == "__main__":
    main()
