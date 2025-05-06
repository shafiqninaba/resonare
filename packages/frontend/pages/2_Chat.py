import streamlit as st
import requests  # Assuming you'll use requests to call the backend
import re  # Import regular expressions module

st.title("Chat with Model")

# Initialize chat history and run_id in session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Format: [{"role": "user/assistant", "content": "..."}]
    st.session_state.displayed_messages = []  # for streamlit to display to bypass the >>> markdown issue
if "run_id" not in st.session_state:
    st.session_state.run_id = ""
if "run_id_locked" not in st.session_state:
    st.session_state.run_id_locked = False

# --- Run ID Input ---
# ... (rest of the Run ID input code remains the same) ...
run_id_input = st.text_input(
    "Enter Run ID",
    value=st.session_state.run_id,
    disabled=st.session_state.run_id_locked,
    key="run_id_input_field",  # Add a key to manage its state if needed elsewhere
)

# Update run_id in session state ONLY if not locked
if not st.session_state.run_id_locked:
    st.session_state.run_id = run_id_input


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
            api_url = "http://localhost:8000/infer"
            # Send the entire message history in the payload
            payload = {
                "run_id": st.session_state.run_id,
                "messages": st.session_state.messages,
            }

            # Set spinner text based on whether it's the first message
            spinner_text = (
                "Downloading model..."
                if is_first_message
                else "Waiting for model response..."
            )

            # Add a spinner while waiting for the backend
            with st.spinner(spinner_text):
                response = requests.post(
                    api_url, json=payload, timeout=120
                )  # Added timeout
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                api_response = response.json()
                assistant_response_content = api_response.get(
                    "response", "Error: No response field found in API JSON."
                )
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
            with st.chat_message("assistant"):
                st.markdown(
                    assistant_response_content or "Error: Empty response received."
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
                    with st.chat_message("assistant"):
                        st.markdown(cleaned_part)
                    # Add each part as a separate message to chat history
                    st.session_state.displayed_messages.append(
                        {"role": "assistant", "content": cleaned_part}
                    )
