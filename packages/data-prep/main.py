#!/usr/bin/env python3
"""
Processes a JSON chat export (e.g., from Telegram) into filtered conversation
sessions, chunked by time and token limits, suitable for further analysis or
LLM fine-tuning.

This script performs the following steps:
1. Loads a model specific tokenizer using HuggingFace Transformers (fallback to TikToken).

2. Loads the raw JSON export from a single file or a directory of files.
    - If a file is specified, it processes that single JSON file, which should contain a list of chats.
    - If a directory is specified, it processes all JSON files within it, which should each contain a single chat.

3. From the loaded raw data, start building `Chat` objects containing `Message` objects
   filtering for textual content and applying the date limit if specified.

3. Chunks messages within each chat into conversation 'blocks' based on time gaps
   (`convo_block_thereshold_secs`) and token counts (`min_tokens_per_block`,
   `max_tokens_per_block`). Discards blocks outside the token range. Discards conversations with no blocks.

4. Merges consecutive messages from the same sender within each block,
   prefixing each original message line with a delimiter (`message_delimiter`).

5. Calculates and logs summary statistics about the processed chats and blocks.

6. Exports the processed data into two JSON / JSONL files:
   - One file containing full chat metadata and all associated blocks.
   - One file containing only the processed blocks, one block per line,
     suitable for ML training pipelines.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import hydra
from loguru import logger
from omegaconf import DictConfig
from rich.progress import (  # TODO: implement rich progress bar for tasks
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from models import Chat, Message
from utils import calculate_chat_stats, load_tokenizer, parse_date_limit


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ---------------------
    # 1) Tokenizer loading
    # ---------------------
    # We need a tokenizer to split chat messages into tokens and obtain token counts, for chunking and filtering:
    #  - Prefer the same tokenizer family (e.g. BPE, SentencePiece, WordPiece) as our target finetuning model for accuracy.
    #  - Use HuggingFace’s AutoTokenizer to load the specific tokenizer for the target model.
    #  - If that fails, fall back to OpenAI’s tiktoken (BPE) for speed and API‑compatibility.
    logger.info(f"Loading tokenizer for model {cfg.model_id} for token counting...")
    tokenizer = load_tokenizer(model_name=cfg.model_id)

    # --------------------------------------------
    # 2) Load raw data from JSON data
    # --------------------------------------------
    # We load the raw JSON export from a single file or a directory of files.
    # if a file is specified, it processes that single JSON file, which should contain a list of individual chats.
    # result.json: {chats: {list: {adam: {messages: []}, "zack": {messages: []}}}}
    # if a directory is specified, it processes all JSON files within it, which should each contain a single chat.
    # result.json: {name: "adam", type: "personal_chat", messages: [{from: "adam", text_entities: [], sticker_emoji: ""}]
    logger.info("Loading chats from raw JSON data...")

    mode = cfg.raw_input.mode.lower()
    if mode not in ["file", "dir"]:
        logger.error(
            f"Invalid raw_input.mode: {mode}; must be 'file' or 'dir', defaulting to 'file'"
        )
        mode = "file"

    # Handle a single export file (e.g., "result.json")
    if mode == "file":
        fp = Path(cfg.raw_input.file)
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
        dd = Path(cfg.raw_input.dir)
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

    logger.info(f"Found {len(raw_chats)} chats to process")

    # --------------------------------------------
    # 3) Build Chat objects
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
    logger.info(
        "Processing chats and building chat objects from loaded raw JSON data..."
    )

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
    # 4) Chunking each chat into conversation 'blocks'
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
    # 5) Merge consecutive messages by sender within each block
    # -------------------------------
    # For each block in each Conversation, we:
    #   • Group consecutive messages from the same sender into one Message
    #   • Prefix every line with the delimiter (e.g. '>>>')
    #   • Separate lines with '\n'
    #   • Keep the timestamp of the first message in each group

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

    # -------------------------------
    # 6) Log summary statistics
    # -------------------------------
    logger.info("Calculating statistics of processed chats...")
    chat_stats = calculate_chat_stats(chats, tokenizer)

    # Define the number of top entries to display
    k = 10

    # Extract and sort the block breakdown by the number of blocks in descending order
    top_k_breakdown = sorted(
        chat_stats["block_breakdown"].items(), key=lambda item: item[1], reverse=True
    )[:k]

    logger.info(
        f"\n"
        f"===== Chat Statistics =====\n"
        f"Total Chats: {chat_stats['num_chats']}\n"
        f"Total Blocks: {chat_stats['num_blocks']}\n\n"
        f"===== Top Chats =====\n"
        + "\n".join(
            [
                f"{rank}. {name}: {count} blocks"
                for rank, (name, count) in enumerate(top_k_breakdown, start=1)
            ]
        )
        + f"\n\n"
        f"===== Token Statistics =====\n"
        f"Min Tokens per Block: {chat_stats['min_tokens_per_block']}\n"
        f"Max Tokens per Block: {chat_stats['max_tokens_per_block']}\n"
        f"Avg Tokens per Block: {chat_stats['avg_tokens_per_block']:.2f}\n\n"
        f"===== Duration Statistics =====\n"
        f"Min Duration per Block (minutes): {chat_stats['min_duration_minutes_per_block']:.2f}\n"
        f"Max Duration per Block (minutes): {chat_stats['max_duration_minutes_per_block']:.2f}\n"
        f"Avg Duration per Block (minutes): {chat_stats['avg_duration_minutes_per_block']:.2f}\n"
    )

    # -------------------------------
    # 7) Export: Processed Chats and Training Blocks as JSONL
    # -------------------------------
    processed_chats_filepath = Path(cfg.processed_chats_filepath)
    training_blocks_filepath = Path(cfg.processed_blocks_filepath)

    # Ensure output directories exist
    processed_chats_filepath.parent.mkdir(parents=True, exist_ok=True)
    training_blocks_filepath.parent.mkdir(parents=True, exist_ok=True)

    # 7.1) Export detailed chat data (metadata + blocks)
    logger.info(
        f"Exporting processed chat data with metadata to: {processed_chats_filepath}"
    )
    try:
        # Collect all chat records into a list
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
                                "timestamp": msg.timestamp.isoformat(),
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

        # Write the list of chat records as a single JSON object
        with processed_chats_filepath.open(
            "w", encoding="utf-8"
        ) as processed_chats_file:
            json.dump(chat_records, processed_chats_file, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"An unexpected error occurred during chat metadata export: {e}")
        raise

    # 7.2) Export blocks only (one block per line, suitable for training)
    logger.info(
        f"Exporting training-ready blocks (JSONL) to: {training_blocks_filepath}"
    )
    try:
        with training_blocks_filepath.open(
            "w", encoding="utf-8"
        ) as training_blocks_file:
            for chat in chats:
                for block in chat.blocks:
                    # TODO: add comments here for expeceted data format, and common keys like role, content
                    # Build the flat list of role/content dicts
                    record = {
                        "messages": [
                            {"role": msg.role, "content": msg.content} for msg in block
                        ]
                    }
                    # Dump one JSON object per line
                    json.dump(
                        record,
                        training_blocks_file,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    training_blocks_file.write("\n")

    except Exception as e:
        logger.error(f"An unexpected error occurred during training block export: {e}")
        raise


if __name__ == "__main__":
    main()
