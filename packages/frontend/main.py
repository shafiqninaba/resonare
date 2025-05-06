"""Resonare – LLM Twin  • Streamlit home page."""

from __future__ import annotations

import os
import textwrap
import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from utils import parse_json_chats, display_summary, poll_job

# ----------------------------------------------------------------------------- #
#  Constants
# ----------------------------------------------------------------------------- #
DATA_PREP_URL: str = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNE_URL: str = os.getenv("FINE_TUNING_URL", "http://fine-tuning:8000")

DEFAULT_MODEL: str = "meta-llama/Llama-3.2-1B"
DEFAULT_PROMPT: str = "You are Ren Hwa, a kind, sensitive and somewhat bubbly guy."
DEFAULT_NAME: str = "Ren Hwa"
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


# ----------------------------------------------------------------------------- #
#  Backend helpers
# ----------------------------------------------------------------------------- #
def submit_data_prep_job(
    chats: List[Dict[str, Any]],
    overrides: Dict[str, Any],
    base_url: str = DATA_PREP_URL,
) -> Optional[str]:
    """Send a preprocessing request and return the `run_id`."""
    try:
        resp = requests.post(
            f"{base_url}/process",
            json={"chats": chats, "overrides": overrides},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("run_id")
    except requests.RequestException as exc:
        st.error(f"Data‑prep submission failed: {exc}")
        return None


def submit_fine_tune_job(run_id: str, base_url: str = FINE_TUNE_URL) -> Optional[str]:
    """Kick off a fine‑tuning job for a given `run_id`."""
    try:
        resp = requests.post(
            f"{base_url}/fine-tune", json={"run_id": run_id}, timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("run_id")
    except requests.RequestException as exc:
        st.error(f"Fine‑tuning submission failed: {exc}")
        return None


# ----------------------------------------------------------------------------- #
#  UI helper blocks
# ----------------------------------------------------------------------------- #
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


# ----------------------------------------------------------------------------- #
#  Main page logic
# ----------------------------------------------------------------------------- #
def main() -> None:
    """Render the Resonare home page."""
    st.set_page_config(page_title="Resonare – LLM Twin", layout="centered")
    st.title("🧠 Resonare | LLM Twin")

    st.markdown("""
        Welcome to **Resonare**, a platform to fine-tune various LLM models on your own chat history  
        (e.g. Telegram, WhatsApp) using **[Unsloth](https://unsloth.ai)**.

        Build a **digital twin** of yourself!
    """)

    # ──────────────────────────────────────────────────────────────────  STEP 1
    st.markdown("## 📥 Step 1 – Export Your Chat Data")
    st.markdown("""
    You can export chat history through 2 ways:
    - From **Telegram > Settings > Advanced > Export Telegram Data**
    - Or export individual chat threads

    **Make sure to select the following options:**
    - Under Chat export settings: Select Personal Chats and Private Groups
    - Under Media export settings, don't need to select any options as we only need the text data
    - For format, Select the **Machine-readable JSON**

    Then, take note of the downloaded JSON file(s) and it's locations.
    """)
    show_export_examples()  # screenshots + JSON snippets

    # ──────────────────────────────────────────────────────────────────  STEP 2
    st.markdown("## 📤 Step 2 – Upload Your JSON")
    files = st.file_uploader(
        "Drop JSON file(s) here (multi‑select allowed)",
        type="json",
        accept_multiple_files=True,
    )

    chats = parse_json_chats(files) if files else []
    if chats:
        st.success(f"{len(chats)} valid chat file(s) loaded.")

    # ──────────────────────────────────────────────────────────────────  STEP 3
    # Configure preprocessing
    st.markdown("## ⚙️ Step 3: Processing Parameters")
    with st.form("params_form"):
        model_id = st.text_input(
            "Model ID",
            value="meta-llama/Llama-3.2-1B",
            help="Choose which LLM model you want to fine-tune on.",
        )
        target_name = st.text_input(
            "Your Name",
            value="Ren Hwa",
            help="This helps filter out messages from you in the chat.",
        )
        system_prompt = st.text_area(
            "System Prompt",
            value="You are Ren Hwa, a kind, sensitive and somewhat bubbly guy.",
            help="Define your personality or writing tone for the twin.",
        )

        date_input = st.date_input(
            "Date Limit (optional)",
            value=None,  # value=datetime.datetime(2020, 5, 17),
            help="Only include messages from this date onward (if set).",
        )
        date_limit = date_input.isoformat() if date_input else None

        convo_block_thereshold_secs = st.number_input(
            "Block time threshold (secs)",
            min_value=0,
            value=3600,
            help="Conversations are split into blocks using this time gap.",
        )
        min_tokens_per_block = st.number_input(
            "Min tokens per block",
            min_value=0,
            value=300,
            help="Minimum number of tokens per chat block.",
        )
        max_tokens_per_block = st.number_input(
            "Max tokens per block",
            min_value=0,
            value=800,
            help="Maximum number of tokens per chat block.",
        )
        message_delimiter = st.text_input(
            "Message delimiter",
            value=">>>",
            help="Used to separate messages in a single block.",
        )

        submitted = st.form_submit_button("🚀 Process")

    # ──────────────────────────────────────────────────────────────────  SUBMIT
    if submitted:
        if not chats:
            st.error("Please upload at least one JSON export first.")
            st.stop()

        overrides = {
            "model_id": model_id,
            "target_name": target_name,
            "system_prompt": system_prompt,
            "date_limit": date_limit,
            "convo_block_thereshold_secs": convo_block_thereshold_secs,
            "min_tokens_per_block": min_tokens_per_block,
            "max_tokens_per_block": max_tokens_per_block,
            "message_delimiter": message_delimiter,
        }

        run_id = submit_data_prep_job(chats, overrides)
        if not run_id:
            st.stop()

        st.success(f"Job queued – **run_id = `{run_id}`**")
        with st.spinner("Processing…"):
            info = poll_job(f"{DATA_PREP_URL}/jobs/{run_id}")

        if info.get("status") != "completed":
            st.error("❌ Pre‑processing failed.")
            st.stop()

        display_summary(info.get("stats", {}))

        # ──────────────────────────────────────────────────────────  FINE‑TUNE
        st.header("🎯 Fine‑tune")
        if st.button("Start Fine‑tuning"):
            tune_id = submit_fine_tune_job(run_id)
            if not tune_id:
                st.stop()

            st.success("Fine‑tuning started")
            with st.spinner("Training…"):
                t_info = poll_job(f"{FINE_TUNE_URL}/jobs/{tune_id}")

            if t_info.get("status") == "completed":
                st.success("🎉 Fine‑tune finished!")
            else:
                st.error("Fine‑tune did not complete.")

    # ───────────────────────────────────────────────────────────────  FOOTER
    st.caption("Resonare © 2025")


if __name__ == "__main__":
    main()
