# main.py

import os
import requests
import streamlit as st
import datetime

from utils import (
    parse_json_chats,
    display_summary,
    poll_job,
)

DATA_PREP_URL = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNING_URL = os.getenv("FINE_TUNING_URL", "http://fine-tuning:8000")


def submit_data_prep_job(
    chats: list, overrides: dict, url: str = DATA_PREP_URL
) -> str | None:
    """
    Submit a data-prep job by sending chats and overrides to the backend.

    Args:
        chats (list): List of parsed chat dictionaries.
        overrides (dict): Dictionary of user-defined parameters.
        url (str): Base URL of the data-prep service.

    Returns:
        str | None: Run ID if successful, else None.
    """
    endpoint = f"{url}/process"
    payload = {"chats": chats, "overrides": overrides or {}}

    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("run_id")
    except requests.RequestException as e:
        st.error(f"Submission failed: {e}")
        return None


def submit_fine_tuning_job(run_id: str, url: str = FINE_TUNING_URL) -> str | None:
    """
    Submit a fine-tuning job using the run ID.

    Args:
        run_id (str): Run ID of the previously processed data.
        url (str): Base URL of the fine-tuning service.

    Returns:
        str | None: Run ID if submission succeeds, else None.
    """
    endpoint = f"{url}/fine-tune"
    payload = {"run_id": run_id}

    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("run_id")
    except requests.RequestException as e:
        st.error(f"Fine-tuning submission failed: {e}")
        return None


def main() -> None:
    """Main entry point for the Resonare LLM Twin Streamlit app."""
    st.set_page_config(page_title="Resonare - LLM Twin", layout="centered")
    st.title("🧠 Resonare: LLM Twin")
    st.markdown("""
        Welcome to **Resonare**, a platform to fine-tune various LLM models on your own chat history  
        (e.g. Telegram, WhatsApp) using **[Unsloth](https://unsloth.ai)**.

        Build a **digital twin** of yourself!
    """)

    # Step 1: Instructions
    st.markdown("## 📥 Step 1: Export Your Chat Data")
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

    st.markdown("### 🧾 Telegram Export Formats")

    st.markdown("""
    Telegram exports can be structured in two ways:

    - **All chats** (multi-chat export): your JSON will contain a top-level `chats.list`
    - **Individual chat**: the file itself is a single `dict` with `messages`

    Both are supported by Resonare.
    """)

    col1, col2 = st.columns(2)

    with col1:
        try:
            st.image(
                "assets/export_all.png",
                caption="Exporting all chats",
                use_container_width=True,
            )
            st.code(
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
                """.strip(),
                language="json",
            )
        except Exception:
            pass

    with col2:
        try:
            st.image(
                "assets/export_individual.png",
                caption="Exporting a single chat",
                use_container_width=True,
            )
            st.code(
                """
    {
    "name": "salt",
    "type": "personal_chat",
    "messages": [
        {"id": 71179, "from": "salt", "text": [...], ...},
        {"id": 71187, "from": "Ren Hwa", "text": "Thx", ...}
    ]
    }
                """.strip(),
                language="json",
            )
        except Exception:
            pass

    # Step 2: Upload
    st.markdown("## 📤 Step 2: Upload Your JSON File(s)")
    st.markdown(
        "Upload your exported JSON file(s) (e.g. `result.json`). Multiple files are supported."
    )
    files = st.file_uploader(
        "Drop Telegram JSON files here", type="json", accept_multiple_files=True
    )
    chats = parse_json_chats(files) if files else []

    if chats:
        st.success(f"{len(chats)} chat(s) loaded.")

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
            value = datetime.datetime(2020, 5, 17),
            help="Only include messages from this date onward (if set).",
        )
        date_limit = date_input.isoformat() if date_input else None

        convo_secs = st.number_input(
            "Block time threshold (secs)",
            min_value=0,
            value=3600,
            help="Conversations are split into blocks using this time gap.",
        )
        min_tokens = st.number_input(
            "Min tokens per block",
            min_value=0,
            value=300,
            help="Minimum number of tokens per chat block.",
        )
        max_tokens = st.number_input(
            "Max tokens per block",
            min_value=0,
            value=800,
            help="Maximum number of tokens per chat block.",
        )
        delimiter = st.text_input(
            "Message delimiter",
            value=">>>",
            help="Used to separate messages in a single block.",
        )

        process_click = st.form_submit_button("Submit for Processing")

    if process_click:
        if not chats:
            st.error("Please upload at least one valid chat JSON.")
            return

        overrides = {
            "model_id": model_id,
            "target_name": target_name,
            "system_prompt": system_prompt,
            "date_limit": date_limit,
            "convo_block_thershold_secs": convo_secs,
            "min_tokens_per_block": min_tokens,
            "max_tokens_per_block": max_tokens,
            "message_delimiter": delimiter,
        }

        run_id = submit_data_prep_job(chats=chats, overrides=overrides)
        if not run_id:
            return

        st.success(
            f"✅ Data processing job submitted — Run ID: `{run_id}`\n\n"
            f"*(This will also be your model ID for fine-tuning.)*"
        )

        with st.spinner("Processing chat data..."):
            job_info = poll_job(f"{DATA_PREP_URL}/jobs/{run_id}")

        if job_info.get("status") == "completed":
            stats = job_info.get("stats", {})
            display_summary(stats)

            st.header("🎯 Fine-Tuning")
            if st.button("Start Fine-Tuning"):
                st.success("Fine-tuning in progress...SHAFIQ MODIFY HERE")
                # submitted = submit_fine_tuning_job(run_id=run_id)
                # if submitted:
                #     st.success(f"Fine-tuning started! Run ID: `{run_id}`")
                #     with st.spinner("Monitoring fine-tuning status..."):
                #         tune_info = poll_job(
                #             f"{FINE_TUNING_URL}/jobs/{run_id}"
                #         )
                #     if tune_info.get("status") == "completed":
                #         st.success("🎉 Fine-tuning completed!")
                #     else:
                #         st.error("❌ Fine-tuning failed or did not complete.")
        else:
            st.error("Data processing failed.")


if __name__ == "__main__":
    main()
