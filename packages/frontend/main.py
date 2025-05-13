"""Resonare – LLM Twin · Streamlit home page."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from src.utils import (
    display_summary,
    parse_json_chats,
    poll_job,
    show_export_examples,
)

DATA_PREP_URL: str = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNE_URL: str = os.getenv("FINE_TUNING_SERVICE_URL", "http://fine-tuning:8000")

DEFAULT_PROMPT: str = "You are Ren Hwa, a kind, sensitive and somewhat bubbly guy."
DEFAULT_NAME: str = "Ren Hwa"


def submit_data_prep_job(
    chats: List[Dict[str, Any]],
    overrides: Dict[str, Any],
    base_url: str = DATA_PREP_URL,
) -> Optional[str]:
    """
    Send a preprocessing request to the data-prep service.

    Args:
        chats: Parsed chat dictionaries.
        overrides: User-defined parameters for preprocessing.
        base_url: Base URL of the data-prep service.

    Returns:
        The run_id on success, or None on failure.
    """
    try:
        response = requests.post(
            f"{base_url}/process",
            json={"chats": chats, "overrides": overrides},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("run_id")
    except requests.RequestException as exc:
        st.error(f"Data-prep submission failed: {exc}")
        return None


def submit_fine_tune_job(
    run_id: str,
    overrides: Dict[str, Any],
    base_url: str = FINE_TUNE_URL,
) -> Optional[str]:
    """
    Send a fine-tuning request to the fine-tune service.

    Args:
        run_id: Identifier from preprocessing step.
        overrides: User-defined parameters for training.
        base_url: Base URL of the fine-tune service.

    Returns:
        The run_id on success, or None on failure.
    """
    try:
        response = requests.post(
            f"{base_url}/fine-tune",
            json={"run_id": run_id, "overrides": overrides},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("run_id")
    except requests.RequestException as exc:
        st.error(f"Fine-tuning submission failed: {exc}")
        return None


def main() -> None:
    """Render the Resonare LLM Twin Streamlit interface."""
    st.set_page_config(page_title="Resonare – LLM Twin", layout="centered")
    st.title("Resonare | LLM Twin")

    # Introduction
    st.markdown(
        """
        Welcome to **Resonare**, a platform to fine-tune various LLM models on your
        own chat history (e.g. Telegram, WhatsApp) using **[Unsloth](
        https://unsloth.ai)**.

        Build a **digital twin** of yourself!
        """
    )

    # Step 1: Export Instructions
    st.markdown("## 📥 Step 1 – Export Your Chat Data")
    st.markdown(
        """
        You can export chat history through two methods:

        - **Telegram → Settings → Advanced → Export Telegram Data**
        - Export an individual chat thread

        **Ensure you select:**
        - Chats: Personal Chats + Private Groups
        - Format: Machine-readable JSON
        - Media: none (text only)
        """
    )
    show_export_examples()

    # Step 2: Upload JSONs
    st.markdown("## 📤 Step 2 – Upload Your JSON")
    files = st.file_uploader(
        "Drop JSON files here", type="json", accept_multiple_files=True
    )
    chats, sender_names = parse_json_chats(files) if files else ([], [])
    if chats:
        st.success(f"{len(chats)} valid chat file(s) loaded.")

    # Step 3: Preprocessing Parameters
    st.markdown("## ⚙️ Step 3 – Processing Parameters")
    with st.form("data_prep_params_form"):
        target_name = st.selectbox(
            "Target Name",
            options=sender_names or [DEFAULT_NAME],
            help="Filter out messages sent by you.",
        )
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_PROMPT,
            help="Define the personality or tone for your twin.",
        )
        date_input = st.date_input(
            "Date Limit (optional)",
            value=None,
            help="Include messages from this date onward.",
        )
        date_limit = date_input.isoformat() if date_input else None

        convo_secs = st.number_input(
            "Block time threshold (secs)",
            min_value=0,
            value=3600,
            help="Split conversations if messages are farther apart than this.",
        )
        min_tokens = st.number_input(
            "Min tokens per block",
            min_value=0,
            value=300,
            help="Minimum tokens in a conversation block.",
        )
        max_tokens = st.number_input(
            "Max tokens per block",
            min_value=0,
            value=800,
            help="Maximum tokens in a conversation block.",
        )
        delimiter = st.text_input(
            "Message delimiter",
            value=">>>",
            help="Prefix for each merged message line.",
        )

        submitted = st.form_submit_button("🚀 Process")

    # Handle preprocessing submission
    if submitted:
        if not chats:
            st.error("Please upload at least one JSON file first.")
            st.stop()

        prep_overrides = {
            "target_name": target_name,
            "system_prompt": system_prompt,
            "date_limit": date_limit,
            "convo_block_thereshold_secs": convo_secs,
            "min_tokens_per_block": min_tokens,
            "max_tokens_per_block": max_tokens,
            "message_delimiter": delimiter,
        }

        run_id = submit_data_prep_job(chats, prep_overrides)
        if not run_id:
            st.stop()
        else:
            st.success(f"🚀 Pre-processing started! Run ID: **{run_id}**. ")

        with st.spinner("Processing…"):
            info = poll_job(f"{DATA_PREP_URL}/jobs/{run_id}")

            if info.get("status") != "completed":
                st.error(
                    f"Data prep job for {run_id} did not complete, error: {info.get('error')}"
                )
                st.stop()
            else:
                st.success(f"Data prep job for {run_id} completed successfully!")
                display_summary(info.get("stats", {}))

            # # Step 4: Fine-Tuning Parameters
            # st.header("🎯 Step 4 – Fine-Tuning")
            # with st.form("finetune_params_form"):
            #     model_options = {
            #         "unsloth/gemma-3-1b-it-unsloth-bnb-4bit": "gemma-3",
            #         "unsloth/Llama-3.2-1B-Instruct-bnb-4bit": "llama-3.2",
            #         "unsloth/Qwen3-1.7B-unsloth-bnb-4bit": "qwen3",
            #     }
            #     model_id = st.selectbox(
            #         "Model ID",
            #         options=list(model_options),
            #         help="Choose a base checkpoint for fine-tuning.",
            #     )
            #     chat_template = model_options[model_id]

            #     st.markdown("### 🪄 LoRA Parameters")
            #     r = st.number_input(
            #         "LoRA Rank (r)",
            #         min_value=1,
            #         value=16,
            #         help="Low-rank matrix dimension (capacity vs speed).",
            #     )
            #     alpha = st.number_input(
            #         "LoRA Alpha",
            #         min_value=1,
            #         value=16,
            #         help="Scaling factor; typically equals r.",
            #     )

            #     st.markdown("### 🏋️ Training Parameters")
            #     batch_size = st.number_input(
            #         "Per-device Batch Size",
            #         min_value=1,
            #         value=1,
            #         help="Examples per GPU batch.",
            #     )
            #     grad_accum = st.number_input(
            #         "Gradient Accumulation Steps",
            #         min_value=1,
            #         value=4,
            #         help="Simulate a larger effective batch size.",
            #     )
            #     warmup = st.number_input(
            #         "Warmup Steps", min_value=0, value=5, help="LR scheduler warmup period."
            #     )
            #     max_steps = st.number_input(
            #         "Max Steps", min_value=1, value=60, help="Total training steps."
            #     )

            #     submitted = st.form_submit_button("🎯 Start Fine-Tuning")

            # if submitted:
            #     tune_overrides = {
            #         "model_id": model_id,
            #         "chat_template": chat_template,
            #         "lora_r": r,
            #         "lora_alpha": alpha,
            #         "per_device_train_batch_size": batch_size,
            #         "gradient_accumulation_steps": grad_accum,
            #         "warmup_steps": warmup,
            #         "max_steps": max_steps,
            #     }

            #     run_id = submit_fine_tune_job(run_id, tune_overrides)
            #     if not run_id:
            #         st.stop()
            #     else:
            #         st.success(
            #             f"🚀 Finetuning started! Run ID: **{run_id}**. "
            #             "Save this for retrieving the trained model for inference later."
            #         )

            with st.spinner("Training…"):
                t_info = poll_job(f"{FINE_TUNE_URL}/jobs/{run_id}")

                if t_info.get("status") != "completed":
                    st.error(
                        f"Fine-tuning job for {run_id} did not complete, error: {t_info.get('error')}"
                    )
                    st.stop()
                else:
                    st.success(f"Finetuning job for {run_id} completed successfully!")

    st.caption("Resonare © 2025")


if __name__ == "__main__":
    main()
