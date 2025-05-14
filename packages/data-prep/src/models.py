#!/usr/bin/env python3
"""
Core data models for processing Telegram chat data.

Defines structured representations for messages, blocks and chat conversations,
supporting tasks like chunking messages into blocks for machine learning training.
Provides a type-safe way to handle and preprocess chat data.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, constr, model_validator


class Message(BaseModel):
    """Represents a single message within a chat.

    Attributes:
        role: Identifier for the message sender ('system', 'user', 'assistant').
        content: Textual content of the message.
        timestamp: Datetime object indicating when the message was sent (optional).
    """

    role: Literal["system", "user", "assistant"]
    content: constr(strip_whitespace=True, min_length=1)  # ✅ Type expression, not call
    timestamp: Optional[datetime] = None


class Block(BaseModel):
    """Represents a contiguous conversation segment for model training.

    A Block contains:
        1. An optional leading system message.
        2. A sequence of user and assistant messages in strict alternation,
           starting with `user` and ending with `assistant`.

    Attributes:
        messages (List[Message]):
            Ordered messages in this block. Must begin with 'system' (optional),
            then 'user', and alternate between 'assistant' and 'user',
            ending with 'assistant'.
    """

    messages: List[Message]

    @model_validator(
        mode="after"
    )  # Ren Hwa: what is the difference between after and before?
    def validate_block_structure(cls, block):
        msgs = block.messages

        if not msgs:
            raise ValueError("Block is empty")

        i = 0

        # Optional system message at the start
        if msgs[0].role == "system":
            i += 1

        if i >= len(msgs):
            raise ValueError("Block must contain more than just a system message")

        # First non-system message must be user
        if msgs[i].role != "user":
            raise ValueError("First message after system must be from 'user'")

        # Alternating pattern from this point
        expected = "assistant"
        for m in msgs[i + 1 :]:
            if m.role != expected:
                raise ValueError(
                    f"Expected '{expected}' but got '{m.role}' in message sequence"
                )
            expected = "user" if expected == "assistant" else "assistant"

        if msgs[-1].role != "assistant":
            raise ValueError("Block must end with an 'assistant' message")

        return block


class Chat(BaseModel):
    """Represents a chat conversation, potentially broken into blocks.

    Attributes:
        contact_name: The name of the contact or group chat.
        type: The type of chat (e.g., personal, group).
        messages: A flat list of all processed messages in chronological order.
        raw_blocks: A list of message blocks. Each block is a list of temporally
                and contextually related messages, chunked according to time
                and token limits. Defaults to an empty list.
        valid_blocks: A list of validated Block models. This is populated after chunking, merging,
                    and validating the blocks. Defaults to an empty list.
    """

    contact_name: str
    type: Literal[
        "personal_chat", "private_group", "private_supergroup", "public_supergroup"
    ]
    messages: List[Message]
    raw_blocks: List[List[Message]] = []  # unmerged, unvalidated message lists
    valid_blocks: List[Block] = []  # validated Block models
