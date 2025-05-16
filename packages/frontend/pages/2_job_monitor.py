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
        st.warning(f"⚠️ Could not reach '{url.split('/')[-1]}' endpoint: {exc}")
        return {}


def main() -> None:
    """
    Resonare Job Dashboard page.

    Shows all of the current user's preprocessing and fine-tuning
    runs and their positions in the queues.
    """
    st.set_page_config(page_title="Resonare – Job Dashboard", layout="wide")
    st.title("Resonare – Job Monitor")
    st.markdown("Track your own data-prep and fine-tuning jobs in real time.")

    # Retrieve user's run IDs from session state
    run_ids: List[str] = st.session_state.get("run_ids", [])
    if not run_ids:
        st.info("You have no jobs yet. Start a run on the home page.")
        return

    # Fetch job statuses
    raw_dp = _safe_json(f"{DATA_PREP_URL}/jobs")
    raw_ft = _safe_json(f"{FINE_TUNE_URL}/jobs")

    # Build DataFrame for filtering
    rows: List[Dict[str, Any]] = []
    for rid, info in raw_dp.items():
        if rid in run_ids:
            rows.append({
                "run_id": rid,
                "type": "data-prep",
                "status": info.get("status", "unknown"),
                "created_at": info.get("created_at", ""),
                "started_at": info.get("started_at", ""),
                "completed_at": info.get("completed_at", ""),
                "error": info.get("error", ""),
            })
    for rid, info in raw_ft.items():
        if rid in run_ids:
            rows.append({
                "run_id": rid,
                "type": "fine-tuning",
                "status": info.get("status", "unknown"),
                "created_at": info.get("created_at", ""),
                "started_at": info.get("started_at", ""),
                "completed_at": info.get("completed_at", ""),
                "error": info.get("error", ""),
            })
    df = pd.DataFrame(rows)

    # Filters
    st.markdown("### 🔍 Filter Jobs")
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Status", options=["queued", "running", "completed", "failed"], key="status_filter"
        )
    with col2:
        type_filter = st.multiselect(
            "Job Type", options=["data-prep", "fine-tuning"], key="type_filter"
        )
    with col3:
        run_id_query = st.text_input("Search Run ID", key="runid_search")

    if status_filter:
        df = df[df["status"].isin(status_filter)]
    if type_filter:
        df = df[df["type"].isin(type_filter)]
    if run_id_query:
        df = df[df["run_id"].str.contains(run_id_query, case=False, na=False)]

    # Job summary table
    st.markdown("### 📋 Your Job Summary")
    if df.empty:
        st.info("No matching jobs.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False), use_container_width=True)

    # Queue positions
    st.markdown("### 📡 Current Queue Positions")
    raw_dp_queue = _safe_json(f"{DATA_PREP_URL}/jobs/queue")
    raw_ft_queue = _safe_json(f"{FINE_TUNE_URL}/queue")

    dp_q = raw_dp_queue.get("jobs", {})
    ft_q = raw_ft_queue.get("jobs", {})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data-Prep Queue")
        dp_rows = []
        for rid, job in dp_q.items():
            # Only show queued (not running)
            if rid in run_ids and job.get("status") == "queued":
                dp_rows.append({
                    "Run ID": rid,
                    "Position": job.get("position_in_queue", 0),
                    "Created At": raw_dp.get(rid, {}).get("created_at", ""),
                })
        if not dp_rows:
            st.info("No data-prep jobs queued.")
        else:
            dp_table = pd.DataFrame(dp_rows).sort_values("Position")
            st.table(dp_table)

    with col2:
        st.subheader("Fine-Tuning Queue")
        ft_rows = []
        for rid, job in ft_q.items():
            if rid in run_ids and job.get("status") == "queued":
                ft_rows.append({
                    "Run ID": rid,
                    "Position": job.get("position_in_queue", 0),
                    "Created At": raw_ft.get(rid, {}).get("created_at", ""),
                })
        if not ft_rows:
            st.info("No fine-tuning jobs queued.")
        else:
            ft_table = pd.DataFrame(ft_rows).sort_values("Position")
            st.table(ft_table)


if __name__ == "__main__":
    main()
