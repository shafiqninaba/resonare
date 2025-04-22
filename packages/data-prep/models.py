#!/usr/bin/env python3
"""
Core data models for processing Telegram chat data.

Defines structured representations for messages and chat conversations,
supporting tasks like chunking messages into blocks for machine learning training.
Provides a type-safe way to handle and preprocess chat data.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel


class Message(BaseModel):
    """Represents a single message within a chat.

    Attributes:
        role: Identifier for the message sender ('system', 'user', 'assistant').
        content: Textual content of the message.
        timestamp: Datetime object indicating when the message was sent (optional)..
    """

    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: Optional[datetime] = None


class Chat(BaseModel):
    """Represents a chat conversation, potentially broken into blocks.

    Attributes:
        contact_name: The name of the contact or group chat.
        type: The type of chat (e.g., personal, group).
        messages: A flat list of all processed messages in chronological order.
        blocks: A list of message blocks. Each block is a list of temporally
                and contextually related messages, chunked according to time
                and token limits. Defaults to an empty list.
    """

    contact_name: str
    type: Literal[
        "personal_chat", "private_group", "private_supergroup", "public_supergroup"
    ]
    messages: List[Message]
    blocks: List[List[Message]] = []
