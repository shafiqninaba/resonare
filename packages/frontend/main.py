from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from src.utils import display_chat_summary, parse_json_chats, poll_job

# Service URLs
DATA_PREP_URL = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNE_URL = os.getenv("FINE_TUNING_SERVICE_URL", "http://unsloth-backend:8000")


def submit_data_prep_job(
    chats: List[Dict[str, Any]],
    target_name: str,
    system_prompt: Optional[str] = None,
    date_limit: Optional[str] = None,
    convo_secs: Optional[int] = None,
    min_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    model_id: Optional[str] = None,
    chat_template: Optional[str] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
    base_url: str = DATA_PREP_URL,
) -> Optional[str]:
    """
    Send a preprocessing request to the data-prep service.

    Only include overrides that are not None.

    Returns the run_id on success, or None on error.
    """
    payload: Dict[str, Any] = {"chats": chats, "target_name": target_name}

    # data-prep options
    if system_prompt:
        payload["system_prompt"] = system_prompt
    if date_limit:
        payload["date_limit"] = date_limit
    if convo_secs is not None:
        payload["convo_block_threshold_secs"] = convo_secs
    if min_tokens is not None:
        payload["min_tokens_per_block"] = min_tokens
    if max_tokens is not None:
        payload["max_tokens_per_block"] = max_tokens

    # fine-tuning options
    if model_id:
        payload["name"] = model_id
    if chat_template:
        payload["chat_template"] = chat_template
    if lora_r is not None:
        payload["r"] = lora_r
    if lora_alpha is not None:
        payload["alpha"] = lora_alpha
    if batch_size is not None:
        payload["per_device_train_batch_size"] = batch_size
    if grad_accum is not None:
        payload["gradient_accumulation_steps"] = grad_accum
    if warmup_steps is not None:
        payload["warmup_steps"] = warmup_steps
    if max_steps is not None:
        payload["max_steps"] = max_steps

    try:
        resp = requests.post(f"{base_url}/jobs/submit", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("run_id")
    except requests.RequestException as exc:
        st.error(f"Data-prep submission failed: {exc}")
        return None


def submit_fine_tune_job(
    run_id: str,
    model_id: Optional[str] = None,
    chat_template: Optional[str] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
    base_url: str = FINE_TUNE_URL,
) -> Optional[str]:
    """
    Send a fine-tuning request to the fine-tune service.

    Only include overrides that are not None.

    Returns the run_id on success, or None on error.
    """
    payload: Dict[str, Any] = {"run_id": run_id}
    if model_id:
        payload["model_id"] = model_id
    if chat_template:
        payload["chat_template"] = chat_template
    if lora_r is not None:
        payload["lora_r"] = lora_r
    if lora_alpha is not None:
        payload["lora_alpha"] = lora_alpha
    if batch_size is not None:
        payload["per_device_train_batch_size"] = batch_size
    if grad_accum is not None:
        payload["gradient_accumulation_steps"] = grad_accum
    if warmup_steps is not None:
        payload["warmup_steps"] = warmup_steps
    if max_steps is not None:
        payload["max_steps"] = max_steps

    try:
        resp = requests.post(f"{base_url}/fine-tune", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("run_id")
    except requests.RequestException as exc:
        st.error(f"Fine-tuning submission failed: {exc}")
        return None


def main() -> None:
    """
    Render the Resonare LLM Twin Streamlit interface.
    """
    st.set_page_config(page_title="Resonare – LLM Twin", layout="centered")
    st.title("Resonare | LLM Twin")
    st.markdown(
        """
        Welcome to Resonare, a platform to fine-tune various LLM models
        on your personal Telegram chat data.

        Create a digital twin of yourself today!
        """
    )

    # Persist past run IDs
    if "run_ids" not in st.session_state:
        st.session_state.run_ids = []

    # --- Step 1: Export Instructions ---
    st.header("📥 Step 1: Export Your Chat Data")
    with st.expander("Click here for export instructions", expanded=False):
        st.markdown(
            """
        **Option A: Export All Chats**
        1. Telegram Desktop → Settings → Advanced → Export Telegram Data  
        2. Select the following options:  
        - **Chats**: Personal chats (Support for groups coming soon)  
        - **Export format**: JSON  
        - **Export media**: None  
        3. Click Export

        **Option B: Export Single Chat**
        1. Open individual chat in Telegram Desktop  
        2. Click on the three dots in the top right corner  
        3. Select **Export chat history**  
        4. Same settings as above  
        5. Repeat per chat
        """
        )

    st.markdown(
        """ 
        After exporting, you’ll typically find the files in:  
        - **Windows**: `C:\\Users\\<username>\\Downloads\\Telegram Desktop\\`  
        - **macOS**: `~/Downloads/Telegram Desktop/`  

        Drag and drop the JSON files into the box below.
        """
    )

    # --- Step 2: Upload ---
    st.header("📤 Step 2: Upload JSON Files")
    files = st.file_uploader(
        "Drop JSON exports here:",
        type="json",
        accept_multiple_files=True,
    )
    chats: List[Dict[str, Any]] = []
    sender_names: List[str] = []
    if files:
        chats, sender_names = parse_json_chats(files)
        if chats:
            st.success(f"Loaded {len(chats)} chat file(s).")

    # --- Step 3: Select Target ---
    st.header("Step 3: Select Target")
    target_name: Optional[str] = None
    if sender_names:
        target_name = st.selectbox(
            "Person to train on:",
            options=sender_names,
            help="Select the name to train your LLM twin on.",
        )
    else:
        st.warning("No valid chat data found. Please upload JSON file(s) first.")

    # --- Advanced Options ---
    st.header("Step 4: Advanced Options")
    with st.expander("Click here for advanced options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            system_prompt = st.text_area(
                "System Prompt (optional)",
                height=100,
                help="Define the personality or tone for your twin.",
            )
            date_limit_date = st.date_input(
                "Start Date (optional)",
                value=None,
                help="Include messages from this date onward.",
            )
            date_limit = date_limit_date.isoformat() if date_limit_date else None
            convo_secs = st.number_input(
                "Block Time Threshold (secs)",
                min_value=0,
                value=None,
                help="Split conversations if messages are farther apart than this.",
            )
            min_tokens = st.number_input(
                "Min Tokens per Block",
                min_value=0,
                value=None,
                help="Minimum tokens in a conversation block.",
            )
            max_tokens = st.number_input(
                "Max Tokens per Block",
                min_value=0,
                value=None,
                help="Maximum tokens in a conversation block.",
            )
        with col2:
            model_map = {
                "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit": "llama-3.1",
                "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit": "llama-3.2",
                "unsloth/llama-3-8b-Instruct-bnb-4bit": "llama-3",
                # unsloth prefix indicate that they are Unsloth dynamic 4-bit quants.
                # These models consume slightly more VRAM than standard BitsAndBytes 4-bit models but offer significantly higher accuracy.
                # "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit": "llama-3.2",
                # "unsloth/Llama-3.2-1B-Instruct-bnb-4bit": "llama-3.2",
                # "unsloth/gemma-3-4b-it-unsloth-bnb-4bit": "gemma-3",
            }
            model_id = st.selectbox(
                "Base Model",
                options=list(model_map.keys()),
                help="Choose a base checkpoint for fine-tuning.",
            )
            chat_template = model_map[model_id]

            lora_r = st.number_input(
                "LoRA Rank (r)",
                min_value=1,
                value=None,
                help="Low-rank matrix dimension (capacity vs speed).",
            )
            lora_alpha = st.number_input(
                "LoRA Alpha",
                min_value=1,
                value=None,
                help="Scaling factor; typically equals r.",
            )
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                value=None,
                help="Examples per GPU batch.",
            )
            grad_accum = st.number_input(
                "Gradient Accumulation Steps",
                min_value=1,
                value=None,
                help="Simulate a larger effective batch size.",
            )
            warmup_steps = st.number_input(
                "Warmup Steps",
                min_value=0,
                value=None,
                help="LR scheduler warmup period.",
            )
            max_steps = st.number_input(
                "Max Training Steps",
                min_value=1,
                value=None,
                help="Total training steps.",
            )

    # --- Submission ---
    if st.button("🚀 Start building!"):
        if not chats:
            st.error("Upload at least one JSON file first.")
        elif not target_name:
            st.error("Select a target name before continuing.")
        else:
            st.info("Queuing data-prep job …")
            run_id = submit_data_prep_job(
                chats=chats,
                target_name=target_name,
                system_prompt=system_prompt,
                date_limit=date_limit,
                convo_secs=convo_secs,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                model_id=model_id,
                chat_template=chat_template,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                batch_size=batch_size,
                grad_accum=grad_accum,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
            )
            if run_id:
                st.success("Job successfully queued.")
                st.markdown(f"""
                **Run ID:**
                ```
                {run_id}
                ```
                Please save your run ID. This will be used to fetch your model later.
                """)
                st.session_state.run_ids.append(run_id)

    # --- Current Job Status ---
    st.header("Current Job Status")
    if st.session_state.run_ids:
        current = st.session_state.run_ids[-1]
        st.markdown(f"**Current Run ID:** {current}")
        # Poll data-prep until done
        with st.spinner(f"Processing data-prep for {current}…", show_time=True):
            while True:
                dp = poll_job(f"{DATA_PREP_URL}/jobs?run_id={current}")
                if dp.get("status") in ("completed", "failed"):
                    break
                time.sleep(2)

        if dp.get("status") == "completed":
            st.success("Data-prep completed!")
            display_chat_summary(dp.get("stats", {}))
        else:
            st.error(f"Data-prep failed: {dp.get('error')}")

        # Poll fine-tune until done
        with st.spinner(f"Running fine-tune for {current}…", show_time=True):
            while True:
                ft = poll_job(f"{FINE_TUNE_URL}/jobs/{current}")
                if ft.get("status") in ("completed", "failed"):
                    break
                time.sleep(2)

        if ft.get("status") == "completed":
            st.success("Fine-tuning completed!")
            st.page_link(
                "pages/1_inference.py",
                label="Click here to start chatting with your trained model",
                use_container_width=True,
            )
        else:
            st.error(f"Fine-tuning failed: {ft.get('error')}")

    else:
        st.info("No jobs submitted yet.")

    st.caption("Resonare © 2025 — Built by Shafiq and Ren Hwa")


if __name__ == "__main__":
    main()
