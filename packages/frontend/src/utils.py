"""Utility functions for the Resonare Streamlit frontend."""

import json
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


def parse_json_chats(
    files,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load and validate chats from uploaded JSON, and collect senders.

    Args:
        files: Streamlit-uploaded file objects.

    Returns:
        A tuple containing:
        - List of validated chat dicts.
        - List of unique sender names sorted by frequency.
    """
    raw_chats: List[Dict[str, Any]] = []
    sender_counter: Counter[str] = Counter()

    for file in files:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            st.error(f"❌ Could not parse {file.name}")
            continue

        # Extract top-level chat list
        if isinstance(data, list):
            chats = [c for c in data if {"name", "messages"} <= c.keys()]
        elif isinstance(data, dict):
            if "chats" in data and "list" in data["chats"]:
                chats = data["chats"]["list"]
            elif {"name", "messages"} <= data.keys():
                chats = [data]
            else:
                chats = []
        else:
            chats = []

        if not chats:
            st.warning(f"⚠️ No valid chat objects in {file.name}")
            continue

        # Count each message sender
        for chat in chats:
            for msg in chat.get("messages", []):
                sender = msg.get("from")
                if sender:
                    sender_counter[sender.strip()] += 1
            raw_chats.append(chat)

    sorted_senders = [name for name, _ in sender_counter.most_common()]
    return raw_chats, sorted_senders


def poll_job(url: str, interval: int = 20) -> Dict[str, Any]:
    """
    Poll a FastAPI job endpoint until it finishes or errors.

    Args:
        url: Full job status URL (e.g. http://.../jobs/{run_id}).
        interval: Seconds between polls.

    Returns:
        The final job status JSON, or an error dict.
    """
    while True:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as exc:
            st.error(f"Error polling job: {exc}")
            return {"status": "error", "error": str(exc)}

        if result.get("status") in ("completed", "failed"):
            return result

        time.sleep(interval)


def display_chat_summary(stats: Dict[str, Any]) -> None:
    """
    Show training-data statistics in a collapsible panel:
      - The raw `stats` JSON
      - A small pie chart of the Top-10 chat block counts

    Args:
        stats: The 'stats' mapping from the data-prep response.
    """
    with st.expander("View training data statistics", expanded=False):
        st.subheader("📋 Raw Statistics")
        # Display the full stats dict as JSON
        st.json(stats)

        # Pie chart of Top-10 block counts
        breakdown = stats.get("block_breakdown", {})
        if not breakdown:
            st.info("No block breakdown available.")
            return

        st.subheader("📊 Top-10 Chat Distribution")
        # Build DataFrame and group “others”
        df = (
            pd.DataFrame(breakdown.items(), columns=["Chat", "Blocks"])
            .sort_values("Blocks", ascending=False)
            .reset_index(drop=True)
        )
        top = df.head(10).copy()
        if len(df) > 10:
            others_sum = int(df["Blocks"].iloc[10:].sum())
            top.loc[len(top)] = ["Others", others_sum]

        # Smaller pie chart
        fig, ax = plt.subplots(figsize=(3, 3))
        wedges, texts, autopcts = ax.pie(
            top["Blocks"],
            labels=top["Chat"],
            autopct=lambda pct: f"{pct:.1f}%",
            startangle=90,
            pctdistance=0.75,
        )
        ax.axis("equal")
        plt.setp(autopcts, size=7)
        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)
