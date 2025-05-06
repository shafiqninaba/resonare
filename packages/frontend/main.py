"""Resonare – LLM Twin  • Streamlit home page."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from utils import display_summary, parse_json_chats, poll_job, show_export_examples

# Constants
DATA_PREP_URL: str = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNE_URL: str = os.getenv("FINE_TUNING_URL", "http://fine-tuning:8000")

DEFAULT_PROMPT: str = "You are Ren Hwa, a kind, sensitive and somewhat bubbly guy."
DEFAULT_NAME: str = "Ren Hwa"


# Endpoints Helper Functions
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


def submit_fine_tune_job(
    run_id: str, overrides: Dict[str, Any], base_url: str = FINE_TUNE_URL
) -> Optional[str]:
    """Kick off a fine‑tuning job for a given `run_id`."""
    try:
        resp = requests.post(
            f"{base_url}/fine-tune",
            json={"run_id": run_id, "overrides": overrides},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("run_id")
    except requests.RequestException as exc:
        st.error(f"Fine‑tuning submission failed: {exc}")
        return None


def main() -> None:
    """Render the Resonare home page."""
    st.set_page_config(page_title="Resonare – LLM Twin", layout="centered")
    st.title("🧠 Resonare | LLM Twin")

    ## Introduction
    st.markdown("""
        Welcome to **Resonare**, a platform to fine-tune various LLM models on your own chat history  
        (e.g. Telegram, WhatsApp) using **[Unsloth](https://unsloth.ai)**.

        Build a **digital twin** of yourself!
    """)

    ## Step 1: Instructions to export chat data
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

    ## Step 2: Upload JSON files
    st.markdown("## 📤 Step 2 – Upload Your JSON")
    files = st.file_uploader(
        "Drop JSON file(s) here (multi‑select allowed)",
        type="json",
        accept_multiple_files=True,
    )
    chats, chat_names = parse_json_chats(files) if files else ([], [])
    if chats:
        st.success(f"{len(chats)} valid chat file(s) loaded.")

    ## Step 3: Configure preprocessing
    st.markdown("## ⚙️ Step 3: Processing Parameters")
    with st.form("data_prep_params_form"):
        target_name = st.selectbox(
            "Target Name",
            options=chat_names if chat_names else ["Ren Hwa"],
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

    # Submit data prep job
    if submitted:
        if not chats:
            st.error("Please upload at least one JSON export first.")
            st.stop()

        overrides = {
            "target_name": target_name,
            "system_prompt": system_prompt,
            "date_limit": date_limit,
            "convo_block_thereshold_secs": convo_block_thereshold_secs,
            "min_tokens_per_block": min_tokens_per_block,
            "max_tokens_per_block": max_tokens_per_block,
            "message_delimiter": message_delimiter,
        }

        run_id = submit_data_prep_job(chats, overrides)
        st.success(
            f"🚀 Pre‑processing started! Your run ID is **{run_id}**. "
            f"Take note of this ID to retrieve the correct model for inference later after fine-tuning.",
        )

        with st.spinner("Processing…"):
            info = poll_job(f"{DATA_PREP_URL}/jobs/{run_id}")

        if info.get("status") != "completed":
            st.error("❌ Pre‑processing failed.")
            st.stop()

        display_summary(info.get("stats", {}))

        ## Step 4: Finetuning job
        st.header("🎯 Step 4: Fine‑tuning")
        with st.form("finetune_params_form"):
            model_options = {
                "unsloth/gemma-3-1b-it-unsloth-bnb-4bit": "gemma-3",
                "unsloth/Llama-3.2-1B-Instruct-bnb-4bit": "llama-3.2",
                "unsloth/Qwen3-1.7B-unsloth-bnb-4bit": "qwen3",
            }
            model_id = st.selectbox(
                "Model ID",
                options=list(model_options.keys()),
                help="Select a base model checkpoint for fine-tuning.",
            )

            chat_template = model_options[model_id]

            # LoRA settings
            st.markdown("### 🪄 LoRA Parameters")
            r = st.number_input(
                "LoRA Rank (r)",
                min_value=1,
                value=16,
                help="Dimensionality of the low-rank matrices. Smaller = faster, lower capacity.",
            )
            alpha = st.number_input(
                "LoRA Alpha",
                min_value=1,
                value=16,
                help="LoRA scaling factor. Typically same as r.",
            )

            # Training settings
            st.markdown("### 🏋️ Training Parameters")
            batch_size = st.number_input(
                "Per-device Batch Size",
                min_value=1,
                value=1,
                help="Number of examples per batch on each device.",
            )
            grad_accum = st.number_input(
                "Gradient Accumulation Steps",
                min_value=1,
                value=4,
                help="Simulate larger batch size by accumulating gradients.",
            )
            warmup_steps = st.number_input(
                "Warmup Steps",
                min_value=0,
                value=5,
                help="Number of warmup steps for the LR scheduler.",
            )
            max_steps = st.number_input(
                "Max Steps",
                min_value=1,
                value=60,
                help="Total training steps (batches × accumulation steps).",
            )
            # Submit fine-tune job
            train_submitted = st.form_submit_button("🎯 Start Fine-Tuning")

        if train_submitted:
            overrides = {
                "model_id": model_id,
                "chat_template": chat_template,
                "lora_r": r,
                "lora_alpha": alpha,
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
            }

            submit_fine_tune_job(run_id, overrides)

            st.success("Fine‑tuning started")
            with st.spinner("Training…"):
                t_info = poll_job(f"{FINE_TUNE_URL}/jobs/{run_id}")

            if t_info.get("status") == "completed":
                st.success("🎉 Fine‑tune finished!")
            else:
                st.error("Fine‑tune did not complete.")

    st.caption("Resonare © 2025")


if __name__ == "__main__":
    main()
