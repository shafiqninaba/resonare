import os
import re
from urllib.parse import urljoin

import requests
import streamlit as st

st.title("Resonare | Chat with your model")

# Initialize chat history and run_id in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Format: [{"role": "user/assistant", "content": "..."}]
    st.session_state.displayed_messages = []  # for streamlit to display to bypass the >>> markdown issue

if "run_id" not in st.session_state:
    st.session_state.run_id = ""

if "run_ids" not in st.session_state:
    st.session_state.run_ids = []  # List to store run IDs

if "run_id_locked" not in st.session_state:
    st.session_state.run_id_locked = False

if "train_metadata" not in st.session_state:
    st.session_state.train_metadata = {"x-amz-meta-target_name": "assistant"}


# --- Run ID Input ---
run_id_input = st.text_input(
    "Enter Run ID",
    value=st.session_state.run_ids[-1] if st.session_state.run_ids else "",
    disabled=st.session_state.run_id_locked,
    key="run_id_input_field",  # Add a key to manage its state if needed elsewhere
    help="32-character string representing the run ID of your model, find it in the job monitor.",
)

# Update run_id in session state ONLY if not locked
if not st.session_state.run_id_locked:
    st.session_state.run_id = run_id_input


# --- Temperature Slider ---
temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.05,
    help="Controls randomness: higher values (e.g. 1.0) make responses more creative, lower values (e.g. 0.1) make them more focused and predictable.",
)

# --- Training Metadata ---
# Display training metadata in a collapsible section
if "train_metadata" in st.session_state:
    with st.expander("Training Metadata", expanded=False):
        st.json(st.session_state.train_metadata)


# Display chat messages from history on app rerun
for message in st.session_state.displayed_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Check if run_id is provided before allowing chat
    if not st.session_state.run_id:
        st.warning("Please enter a Run ID before starting the chat.")
    else:
        # Determine if this is the first message exchange
        is_first_message = not st.session_state.messages

        # Lock the run_id input after the first message is submitted
        if is_first_message:
            st.session_state.run_id_locked = True

        # Add user message to chat history FIRST
        st.session_state.messages.append({"role": "user", "content": ">>> " + prompt})
        # Add the user message to displayed messages to avoid markdown issues
        st.session_state.displayed_messages.append({"role": "user", "content": prompt})

        # Display user message immediately (after adding to history)
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Backend API Call ---
        assistant_response_content = (
            None  # Variable to hold the actual response string or error
        )
        is_error = False
        try:
            # Assuming your FastAPI backend is running at http://localhost:8000
            INFERENCE_URL = os.getenv(
                "INFERENCE_URL", "http://unsloth-backend:8000/infer/"
            )

            # join the inference URL with the endpoint
            api_url = urljoin(INFERENCE_URL, "/infer")
            # Send the entire message history in the payload
            payload = {
                "run_id": st.session_state.run_id.strip(),
                "messages": st.session_state.messages,
                "temperature": temperature,  # Add temperature to payload
            }

            # Set spinner text based on whether it's the first message
            spinner_text = (
                "Loading Model..."
                if is_first_message
                else "Waiting for model response..."
            )

            # Add a spinner while waiting for the backend
            with st.spinner(spinner_text):
                response = requests.post(
                    api_url, json=payload, timeout=150,
                )  # Added timeout
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                api_response = response.json()
                assistant_response_content = api_response.get(
                    "response", "Error: No response field found in API JSON."
                )
                train_metadata = api_response.get("metadata", {})
                st.session_state.train_metadata = train_metadata
                if (
                    "Error:" in assistant_response_content
                ):  # Check if the content itself is an error message
                    is_error = True

        except requests.exceptions.Timeout:
            assistant_response_content = "Error: The request to the backend timed out."
            is_error = True
        except requests.exceptions.ConnectionError:
            assistant_response_content = (
                "Error: Could not connect to the backend. Is it running?"
            )
            is_error = True
        except requests.exceptions.RequestException as e:
            # Try to get more detail from the response if it's an HTTP error
            error_detail = (
                f"Status Code: {e.response.status_code}. Response: {e.response.text}"
                if e.response
                else str(e)
            )
            assistant_response_content = f"Error calling backend: {error_detail}"
            st.error(assistant_response_content)  # Show error prominently
            is_error = True
        except Exception as e:
            assistant_response_content = f"An unexpected error occurred: {e}"
            st.error(assistant_response_content)  # Show error prominently
            is_error = True
        # --- End API Call ---

        # Display assistant response(s)
        if is_error or not assistant_response_content:
            # Display the error as a single message
            meta = st.session_state.train_metadata
            role = meta.get("x-amz-meta-target_name", "assistant")
            with st.chat_message(role):
                st.markdown(
                    f"{role}: {assistant_response_content}"
                    or "Error: Empty response received."
                )
            # Add the single error message to history
            # Check if the error message is already the last message to avoid duplicates on rerun
            if not st.session_state.messages or st.session_state.messages[-1].get(
                "content"
            ) != (assistant_response_content or "Error: Empty response received."):
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_response_content
                        or "Error: Empty response received.",
                    }
                )
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response_content}
            )
            # Split the response by ">>> " and process each part
            # Use regex split to handle potential variations in spacing and leading/trailing newlines
            response_parts = re.split(r"\s*>>>\s*", assistant_response_content)
            for part in response_parts:
                cleaned_part = (
                    part.strip()
                )  # Remove leading/trailing whitespace/newlines
                if cleaned_part:  # Only display and add non-empty parts
                    with st.chat_message(train_metadata["x-amz-meta-target_name"]):
                        st.markdown(
                            f"{train_metadata['x-amz-meta-target_name']}: {cleaned_part}"
                        )
                    # Add each part as a separate message to chat history
                    st.session_state.displayed_messages.append(
                        {
                            "role": train_metadata["x-amz-meta-target_name"],
                            "content": f"{train_metadata['x-amz-meta-target_name']}: {cleaned_part}",
                        }
                    )
