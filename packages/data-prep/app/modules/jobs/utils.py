#!/usr/bin/env python3
"""
Utility functions for processing Telegram chat data.

This module provides helper functions for:
- Loading tokenizers (HuggingFace or TikToken) for text processing.
- Parsing date limits to filter messages.
- Calculating statistics for processed chats and their blocks.

These utilities support preprocessing tasks for machine learning pipelines.
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Union

import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from .models import Chat

# Define type alias for tokenizer union
AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast, tiktoken.Encoding]

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str) -> AnyTokenizer:
    """Loads the specified tokenizer, falling back to TikToken.

    Args:
        model_name: The HuggingFace model identifier or path.

    Returns:
        An instance of a HuggingFace tokenizer or a TikToken encoding.

    Raises:
        RuntimeError: If no tokenizer could be loaded.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_bos_token=False,
            trust_remote_code=True,
            use_fast=True,
        )
        logger.info(f"Loaded HuggingFace tokenizer: {model_name}")
        return tokenizer
    except Exception as e:
        logger.warning(
            f"Failed to load HuggingFace tokenizer '{model_name}' "
            f"({e}); defaulting back to tiktoken."
        )
        try:
            # "o200k_base" is a common TikToken encoding used by recent OpenAI models.
            # Adjust if your target model aligns better with a different encoding.
            tokenizer = tiktoken.get_encoding("o200k_base")
            logger.info("Loaded tiktoken tokenizer with encoding: o200k_base")
            return tokenizer
        except Exception as tiktoken_error:
            logger.error(
                f"Failed to load tiktoken ({tiktoken_error}); "
                "cannot proceed without a tokenizer."
            )
            raise RuntimeError("No valid tokenizer available.") from tiktoken_error


def parse_date_limit(date_limit_str: Optional[str]) -> Optional[datetime]:
    """Parses the ISO format date string into a datetime object.

    Args:
        date_limit_str: The date string in ISO format (YYYY-MM-DD HH:MM:SS).

    Returns:
        A datetime object if parsing is successful and the date is not in the
        future, otherwise None.
    """
    if not date_limit_str:
        return None

    try:
        parsed_date_limit = datetime.fromisoformat(date_limit_str)
        # Ensure the date limit is not in the future
        if parsed_date_limit <= datetime.now():
            logger.info(
                f"Applying date limit: Messages before "
                f"{parsed_date_limit.isoformat()} will be ignored."
            )
            return parsed_date_limit
        else:
            logger.info("Provided date_limit is in the future; ignoring date filter.")
            return None
    except ValueError:
        logger.warning(
            f"Invalid date_limit format ('{date_limit_str}'); "
            "expected ISO format (e.g., YYYY-MM-DD or "
            "YYYY-MM-DD HH:MM:SS). Skipping date filter."
        )
        return None


def calculate_chat_stats(
    chats: List[Chat], tokenizer: AnyTokenizer
) -> Dict[str, Union[int, float, Dict[str, int], None]]:
    """
    Calculates summary statistics about the processed chats and their blocks.

    Args:
        chats: A list of processed Chat objects containing messages and blocks.
        tokenizer: The tokenizer instance used for processing.

    Returns:
        A dictionary containing statistics:
        - num_chats: Total number of chats processed.
        - num_blocks: Total number of valid blocks created across all chats.
        - block_breakdown: A dictionary mapping contact_name to num_blocks for that chat.
        - min_tokens_per_block: Minimum token count in a block.
        - max_tokens_per_block: Maximum token count in a block.
        - avg_tokens_per_block: Average token count per block.
        - min_duration_minutes_per_block: Minimum duration (in minutes) of a block.
        - max_duration_minutes_per_block: Maximum duration (in minutes) of a block.
        - avg_duration_minutes_per_block: Average duration (in minutes) per block.
        Returns None for metrics that cannot be calculated (e.g., if no blocks).
    """
    stats: Dict[str, Union[int, float, Dict[str, int], None]] = {
        "num_chats": 0,
        "num_blocks": 0,
        "block_breakdown": {},  # Initialize as an empty dict
        "min_tokens_per_block": None,
        "max_tokens_per_block": None,
        "avg_tokens_per_block": None,
        "min_duration_minutes_per_block": None,
        "max_duration_minutes_per_block": None,
        "avg_duration_minutes_per_block": None,
    }

    if not chats:
        return stats  # Return early if there are no chats

    stats["num_chats"] = len(chats)
    stats["num_blocks"] = sum(len(chat.valid_blocks) for chat in chats)
    stats["block_breakdown"] = {
        chat.contact_name: len(chat.valid_blocks) for chat in chats
    }
    all_blocks = [block for chat in chats for block in chat.valid_blocks]

    # --- Calculate Token Stats ---
    tokens_per_block_list = [
        sum(
            len(tokenizer.encode(msg.content)) for msg in block.messages
        )  # Use tokenizer.encode directly
        for block in all_blocks
    ]
    stats["min_tokens_per_block"] = min(tokens_per_block_list)
    stats["max_tokens_per_block"] = max(tokens_per_block_list)
    stats["avg_tokens_per_block"] = statistics.mean(tokens_per_block_list)

    # --- Calculate Duration Stats ---
    durations_minutes_list = [
        (
            block.messages[-1].timestamp
            - (
                block.messages[1].timestamp
                if block.messages[0].role == "system"
                else block.messages[0].timestamp
            )
        ).total_seconds()
        / 60.0
        for block in all_blocks
    ]
    stats["min_duration_minutes_per_block"] = min(durations_minutes_list)
    stats["max_duration_minutes_per_block"] = max(durations_minutes_list)
    stats["avg_duration_minutes_per_block"] = statistics.mean(durations_minutes_list)

    return stats
