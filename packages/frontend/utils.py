import json
import time
import requests
from typing import List, Dict, Optional, Any
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def parse_json_chats(files) -> List[Dict[str, Any]]:
    """
    Load and validate chat objects from uploaded JSON files.

    Args:
        files: Uploaded file objects from Streamlit.

    Returns:
        List[Dict[str, Any]]: Validated chat dictionaries.
    """
    raw_chats = []
    for file in files:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            st.error(f"Cannot parse {file.name}")
            continue

        chats = []
        if isinstance(data, list):
            chats = [c for c in data if {"name", "messages"} <= c.keys()]
        elif isinstance(data, dict):
            if "chats" in data and "list" in data["chats"]:
                chats = data["chats"]["list"]
            elif {"name", "messages"} <= data.keys():
                chats = [data]
        if not chats:
            st.warning(f"No valid chat objects found in {file.name}")
            continue
        raw_chats.extend(chats)

    return raw_chats


def poll_job(url: str, interval: int = 10) -> Dict[str, Any]:
    """
    Poll a job endpoint until completion or failure.

    Args:
        url (str): Endpoint URL to poll.
        interval (int): Time between polls in seconds.

    Returns:
        Dict[str, Any]: Job status response.
    """
    while True:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            result = resp.json()
        except requests.RequestException as e:
            st.error(f"Error polling job status: {e}")
            return {"status": "error", "error": str(e)}
        if not result:
            return {"status": "error", "error": "Failed to poll job status."}
        status = result.get("status")
        if status in ("completed", "failed"):
            return result
        time.sleep(interval)


def display_summary(stats: Dict[str, Any]) -> None:
    """
    Render headline metrics plus a Top‑10 pie chart of block distribution.

    Args:
        stats: Dictionary produced by the data‑prep backend.
    """
    st.subheader("📊 Summary Statistics")

    # ── headline numbers ─────────────────────────────────────────────
    cols = st.columns(3)
    cols[0].metric("Chats", stats.get("num_chats", 0))
    cols[0].metric("Blocks", stats.get("num_blocks", 0))
    cols[1].metric("Min tokens", stats.get("min_tokens_per_block", 0))
    cols[1].metric("Max tokens", stats.get("max_tokens_per_block", 0))
    cols[2].metric("Avg tokens", round(stats.get("avg_tokens_per_block", 0), 2))

    cols2 = st.columns(3)
    cols2[0].metric(
        "Min dur (min)", round(stats.get("min_duration_minutes_per_block", 0), 2)
    )
    cols2[1].metric(
        "Max dur (min)", round(stats.get("max_duration_minutes_per_block", 0), 2)
    )
    cols2[2].metric(
        "Avg dur (min)", round(stats.get("avg_duration_minutes_per_block", 0), 2)
    )

    # ── block breakdown pie ‑‑ top‑10 ‑‑──────────────────────────────
    breakdown = stats.get("block_breakdown", {})
    if not breakdown:
        return

    st.markdown("### 🥧 Block Distribution (top‑10 chats)")

    df = (
        pd.DataFrame(breakdown.items(), columns=["Chat", "Blocks"])
        .sort_values("Blocks", ascending=False)
        .reset_index(drop=True)
    )

    top_df = df.iloc[:10].copy()
    if len(df) > 10:  # fold remainder
        other_sum = df["Blocks"].iloc[10:].sum()
        top_df.loc[len(top_df)] = ["Others", other_sum]

    total_blocks = top_df["Blocks"].sum()
    top_df["Percent"] = 100 * top_df["Blocks"] / total_blocks

    # ----- plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        top_df["Blocks"],
        labels=top_df["Chat"],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.85,
    )
    ax.axis("equal")  # perfect circle
    plt.setp(autotexts, size=9, weight="bold")
    st.pyplot(fig)
