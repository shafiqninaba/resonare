"""
Resonare – Job Dashboard

Streamlit page that shows all data-prep and fine-tuning jobs plus their
queues, even when the backend returns zero jobs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st


DATA_PREP_URL = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNE_URL = os.getenv("FINE_TUNING_SERVICE_URL", "http://fine-tuning:8000")


def _safe_json(url: str, timeout: int = 15) -> Dict[str, Any]:
    """
    GET JSON from the given URL, returning {} on any failure.

    Args:
        url: Full endpoint (e.g. ".../jobs" or ".../queue").
        timeout: Seconds to wait for a response.

    Returns:
        Parsed JSON or {} on error.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json() or {}
    except Exception as exc:
        st.warning(f"⚠️ Could not reach `{url.split('/')[-1]}`: {exc}")
        return {}


def main() -> None:
    """Entry point for the Resonare Job Dashboard."""
    st.set_page_config(page_title="Resonare – Job Dashboard", layout="wide")
    st.title("Resonare – Job Monitor")
    st.markdown("Track all preprocessing and fine-tuning jobs in real time.")

    # Fetch job statuses and queues
    dp_jobs = _safe_json(f"{DATA_PREP_URL}/jobs", timeout=20)
    ft_jobs = _safe_json(f"{FINE_TUNE_URL}/jobs", timeout=20)
    dp_queue = _safe_json(f"{DATA_PREP_URL}/queue")
    ft_queue = _safe_json(f"{FINE_TUNE_URL}/queue")

    # ── Build combined DataFrame for filtering ────────────────────────────
    rows: List[Dict[str, Any]] = []

    for rid, info in dp_jobs.items():
        rows.append(
            {
                "run_id": rid,
                "type": "data-prep",
                "status": info.get("status", "unknown"),
                "created_at": info.get("created_at", ""),
                "started_at": info.get("started_at", ""),
                "completed_at": info.get("completed_at", ""),
                "error": info.get("error", ""),
                "stats": info.get("stats", {}),
            }
        )
    for rid, info in ft_jobs.items():
        rows.append(
            {
                "run_id": rid,
                "type": "fine-tuning",
                "status": info.get("status", "unknown"),
                "created_at": info.get("created_at", ""),
                "started_at": info.get("started_at", ""),
                "completed_at": info.get("completed_at", ""),
                "error": info.get("error", ""),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "type",
            "status",
            "created_at",
            "started_at",
            "completed_at",
            "error",
            "stats",
        ],
    )

    # Filters
    st.markdown("### 🔍 Filter Jobs")
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Status",
            ["queued", "running", "completed", "failed"],
            key="status_filter",
        )
    with col2:
        type_filter = st.multiselect(
            "Job Type",
            ["data-prep", "fine-tuning"],
            key="type_filter",
        )
    with col3:
        run_id_query = st.text_input("Search Run ID", key="runid_search")

    if status_filter:
        df = df[df["status"].isin(status_filter)]
    if type_filter:
        df = df[df["type"].isin(type_filter)]
    if run_id_query:
        df = df[df["run_id"].str.contains(run_id_query, case=False, na=False)]

    # Job Table
    if df.empty:
        st.info("No matching jobs.")
    else:
        st.dataframe(
            df.sort_values("created_at", ascending=False),
            use_container_width=True,
        )

    # Job Panes
    st.markdown("## All Jobs")

    # Data-Prep Jobs
    if st.checkbox("Data-Prep Jobs", value=True):
        if not dp_jobs:
            st.info("No data-prep jobs yet.")
        for rid, info in sorted(
            dp_jobs.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        ):
            label = f"[data-prep] {rid} — {info.get('status', '-')}"
            with st.expander(label):
                st.json(info)

    # Fine-Tuning Jobs
    if st.checkbox("Fine-Tuning Jobs", value=True):
        if not ft_jobs:
            st.info("No fine-tuning jobs yet.")
        for rid, info in sorted(
            ft_jobs.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        ):
            label = f"[fine-tuning] {rid} — {info.get('status', '-')}"
            with st.expander(label):
                st.json(info)

    # Queues
    st.markdown("## Active Queues")
    q1, q2 = st.columns(2)

    # Data-Prep Queue
    with q1:
        st.subheader("Data-Prep Queue")
        jobs = [
            job
            for job in dp_queue.get("jobs", {}).values()
            if job.get("status") in {"queued", "running"}
        ]
        jobs.sort(key=lambda j: j.get("position_in_queue", 0))
        st.write(f"**Queue size:** {dp_queue.get('queue_size', 0)}")
        if not jobs:
            st.info("No data-prep jobs running or queued.")
        for job in jobs:
            st.json(job)

    # Fine-Tuning Queue
    with q2:
        st.subheader("Fine-Tuning Queue")
        jobs = [
            job
            for job in ft_queue.get("jobs", {}).values()
            if job.get("status") in {"queued", "running"}
        ]
        jobs.sort(key=lambda j: j.get("position_in_queue", 0))
        st.write(f"**Queue size:** {ft_queue.get('queue_size', 0)}")
        if not jobs:
            st.info("No fine-tuning jobs running or queued.")
        for job in jobs:
            st.json(job)


if __name__ == "__main__":
    main()
