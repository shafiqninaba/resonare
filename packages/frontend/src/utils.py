"""Utility functions for the Resonare Streamlit frontend."""

import json
import textwrap
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


def poll_job(url: str, interval: int = 10) -> Dict[str, Any]:
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


def display_summary(stats: Dict[str, Any]) -> None:
    """
    Show summary metrics and a Top-10 pie chart of block counts.

    Args:
        stats: The 'stats' mapping from the data-prep response.
    """
    st.subheader("📊 Summary Statistics")

    # High-level metrics
    cols = st.columns(3)
    cols[0].metric("Chats", stats.get("num_chats", 0))
    cols[0].metric("Blocks", stats.get("num_blocks", 0))
    cols[1].metric("Min tokens", stats.get("min_tokens_per_block", 0))
    cols[1].metric("Max tokens", stats.get("max_tokens_per_block", 0))
    cols[2].metric("Avg tokens", round(stats.get("avg_tokens_per_block", 0), 2))

    cols2 = st.columns(3)
    cols2[0].metric(
        "Min dur (min)", round(stats.get("min_duration_minutes_per_block", 0), 2)
    )
    cols2[1].metric(
        "Max dur (min)", round(stats.get("max_duration_minutes_per_block", 0), 2)
    )
    cols2[2].metric(
        "Avg dur (min)", round(stats.get("avg_duration_minutes_per_block", 0), 2)
    )

    # Pie chart
    breakdown = stats.get("block_breakdown", {})
    if not breakdown:
        return

    st.markdown("### 🥧 Block Distribution (Top 10)")

    df = (
        pd.DataFrame(breakdown.items(), columns=["Chat", "Blocks"])
        .sort_values("Blocks", ascending=False)
        .reset_index(drop=True)
    )

    top_df = df.head(10).copy()
    if len(df) > 10:
        others = df["Blocks"].iloc[10:].sum()
        top_df.loc[len(top_df)] = ["Others", others]

    values = top_df["Blocks"].tolist()
    labels = top_df["Chat"].tolist()

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        pctdistance=0.85,
    )
    ax.axis("equal")
    plt.setp(autotexts, size=9, weight="bold")
    st.pyplot(fig)
