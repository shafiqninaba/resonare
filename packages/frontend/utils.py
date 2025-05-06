import json
import textwrap
import time
from collections import Counter
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


def parse_json_chats(files) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Load and validate chat objects from uploaded JSON files,
    and extract unique sender names from all messages.

    Args:
        files: Uploaded file objects from Streamlit.

    Returns:
        Tuple:
            - List of validated chat dicts
            - List of unique sender names, sorted by frequency
    """
    raw_chats = []
    sender_counter = Counter()

    for file in files:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            st.error(f"❌ Could not parse {file.name}")
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
            st.warning(f"⚠️ No valid chat objects found in {file.name}")
            continue

        for chat in chats:
            messages = chat.get("messages", [])
            for msg in messages:
                sender = msg.get("from")
                if sender:
                    sender = sender.strip()
                    sender_counter[sender] += 1

            raw_chats.append(chat)

    sorted_senders = [name for name, _ in sender_counter.most_common()]
    return raw_chats, sorted_senders


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
    def autopct_format(pct, all_vals):
        absolute = int(round(pct * sum(all_vals) / 100.0))
        return f"{pct:.1f}%\n({absolute})"

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        top_df["Blocks"],
        labels=top_df["Chat"],
        autopct=lambda pct: autopct_format(pct, top_df["Blocks"]),
        startangle=90,
        pctdistance=0.85,
    )
    ax.axis("equal")  # perfect circle
    plt.setp(autotexts, size=9, weight="bold")
    st.pyplot(fig)


# UI Helper Functions
def show_export_examples() -> None:
    """Visualise Telegram export formats (all‑chat vs single‑chat)."""
    st.markdown("### 🧾 Telegram Export Formats")

    st.markdown("""
    Telegram exports can be structured in two ways:

    - **All chats** (multi-chat export): your JSON will contain a top-level `chats.list`
    - **Individual chat**: the file itself is a single `dict` with `messages`

    Both are supported by Resonare.
    """)

    col1, col2 = st.columns(2, gap="medium")

    JSON_ALL_CHATS: str = textwrap.dedent(
        """
        {
        "about": "...",
        "chats": {
            "about": "...",
            "list": [
            {
                "name": "salt",
                "messages": [
                {"id": 71179, "from": "salt", "text": [...], ...},
                {"id": 71187, "from": "Ren Hwa", "text": "Thx", ...}
                ]
            }
            ]
        }
        }
    """
    ).strip()

    JSON_SINGLE_CHAT: str = textwrap.dedent(
        """
        {
        "name": "salt",
        "type": "personal_chat",
        "messages": [
            {"id": 71179, "from": "salt", "text": [...], ...},
            {"id": 71187, "from": "Ren Hwa", "text": "Thx", ...}
        ]
        }
    """
    ).strip()

    with col1:
        st.image(
            "assets/export_all.png", caption="All‑chat export", use_container_width=True
        )
        st.code(JSON_ALL_CHATS, language="json")

    with col2:
        st.image(
            "assets/export_individual.png",
            caption="Single‑chat export",
            use_container_width=True,
        )
        st.code(JSON_SINGLE_CHAT, language="json")
