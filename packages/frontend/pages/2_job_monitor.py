import os
import streamlit as st
import pandas as pd
import requests
from typing import Tuple, Dict, Any

DATA_PREP_URL = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNING_URL = os.getenv("FINE_TUNING_URL", "http://fine-tuning:8000")


def get_job_statuses() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch job statuses from both services."""
    try:
        dp_jobs = requests.get(f"{DATA_PREP_URL}/jobs", timeout=20).json()
    except Exception:
        dp_jobs = {}

    try:
        ft_jobs = requests.get(f"{FINE_TUNING_URL}/jobs", timeout=20).json()
    except Exception:
        ft_jobs = {}

    return dp_jobs, ft_jobs


def get_queues() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetch both queues."""
    try:
        dp_queue = requests.get(f"{DATA_PREP_URL}/queue", timeout=15).json()
    except Exception:
        dp_queue = {}

    try:
        ft_queue = requests.get(f"{FINE_TUNING_URL}/queue", timeout=15).json()
    except Exception:
        ft_queue = {}

    return dp_queue, ft_queue


def display_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Display status and job type filters."""
    st.markdown("### 🔍 Filter Jobs")
    status_list = ["queued", "running", "completed", "failed"]
    job_types = df["type"].unique().tolist()

    selected_status = st.multiselect("Filter by status", status_list)
    selected_type = st.multiselect("Filter by job type", job_types)
    run_id_query = st.text_input("Search Run ID")

    if selected_status:
        df = df[df["status"].isin(selected_status)]
    if selected_type:
        df = df[df["type"].isin(selected_type)]
    if run_id_query:
        df = df[df["run_id"].str.contains(run_id_query)]

    return df


def display_all_jobs(dp_jobs: Dict[str, Any], ft_jobs: Dict[str, Any]) -> None:
    """Display collapsible sections for each job type."""
    st.markdown("## 📋 All Jobs")

    if st.checkbox("Show Data-Prep Jobs", value=True):
        for run_id, info in sorted(
            dp_jobs.items(), key=lambda x: x[1].get("created_at", ""), reverse=True
        ):
            with st.expander(f"[data-prep] {run_id} — {info.get('status')}"):
                st.json(info)

    if st.checkbox("Show Fine-Tuning Jobs", value=True):
        for run_id, info in sorted(
            ft_jobs.items(), key=lambda x: x[1].get("created_at", ""), reverse=True
        ):
            with st.expander(f"[fine-tuning] {run_id} — {info.get('status')}"):
                st.json(info)


def display_queues(dp_queue: Dict[str, Any], ft_queue: Dict[str, Any]) -> None:
    """Display sorted running/queued queues."""
    st.markdown("## 📦 Active Queues")
    col1, col2 = st.columns(2)

    def render_queue(queue: Dict[str, Any]) -> None:
        jobs = {
            rid: data
            for rid, data in queue.get("jobs", {}).items()
            if data.get("status") in ("queued", "running")
        }
        jobs = sorted(jobs.items(), key=lambda x: x[1].get("position_in_queue", 0))

        st.write(f"**Queue size:** {queue.get('queue_size', 0)}")
        for run_id, job in jobs:
            st.json(job)

    with col1:
        st.subheader("Data-Prep Queue")
        render_queue(dp_queue)

    with col2:
        st.subheader("Fine-Tuning Queue")
        render_queue(ft_queue)


def main() -> None:
    """Main entry point for the Resonare job dashboard."""
    st.set_page_config(page_title="Resonare: Job Dashboard", layout="wide")

    st.title("📊 Resonare: Job Monitor")
    st.markdown("Use this dashboard to track all preprocessing and fine-tuning jobs.")

    dp_jobs, ft_jobs = get_job_statuses()
    dp_queue, ft_queue = get_queues()

    combined = []
    for run_id, info in dp_jobs.items():
        combined.append({"run_id": run_id, "type": "data-prep", **info})
    for run_id, info in ft_jobs.items():
        combined.append({"run_id": run_id, "type": "fine-tuning", **info})

    df = pd.DataFrame(combined)
    if not df.empty:
        df = display_filters(df)
        st.dataframe(
            df.sort_values("created_at", ascending=False), use_container_width=True
        )

    display_all_jobs(dp_jobs, ft_jobs)
    display_queues(dp_queue, ft_queue)

    st.markdown("### 🔁 Auto-refresh (optional)")
    st.caption(
        "Use your browser or app to manually refresh. Auto-refresh was removed to avoid bugs in Streamlit."
    )


if __name__ == "__main__":
    main()
