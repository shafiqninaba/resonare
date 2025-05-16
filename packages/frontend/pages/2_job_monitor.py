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
        st.warning(f"⚠️ Could not reach '{url}': {exc}")
        return {}


def main() -> None:
    """
    Resonare Job Dashboard page.

    Shows only the current user's data-prep and fine-tune runs,
    pulled from session_state['run_ids'].
    """
    st.set_page_config(page_title="Resonare – Job Dashboard", layout="wide")
    st.title("Resonare – Job Monitor")
    st.markdown("Track your own preprocessing and fine-tuning jobs.")

    # Retrieve user's run IDs
    run_ids: List[str] = st.session_state.get("run_ids", [])
    if not run_ids:
        st.info("You have no jobs yet. Start a run on the home page.")
        return

    # Fetch all jobs from backend
    raw_dp = _safe_json(f"{DATA_PREP_URL}/jobs")
    raw_ft = _safe_json(f"{FINE_TUNE_URL}/jobs")
    raw_dp_queue = _safe_json(f"{DATA_PREP_URL}/jobs/queue")
    raw_ft_queue = _safe_json(f"{FINE_TUNE_URL}/queue")

    # Filter to user's runs
    dp_jobs = {rid: info for rid, info in raw_dp.items() if rid in run_ids}
    ft_jobs = {rid: info for rid, info in raw_ft.items() if rid in run_ids}

    # Build DataFrame for summary table
    rows: List[Dict[str, Any]] = []
    for rid in run_ids:
        # Data-prep row
        dp = dp_jobs.get(rid, {})
        rows.append({
            "run_id": rid,
            "step": "data-prep",
            "status": dp.get("status", "-"),
            "created_at": dp.get("created_at", "-"),
            "error": dp.get("error", ""),
            "stats": dp.get("stats", {}),
        })
        # Fine-tune row
        ft = ft_jobs.get(rid, {})
        rows.append({
            "run_id": rid,
            "step": "fine-tuning",
            "status": ft.get("status", "-"),
            "created_at": ft.get("created_at", "-"),
            "error": ft.get("error", ""),
            "stats": {},
        })

    df = pd.DataFrame(rows)

    # Display summary table
    st.markdown("### 🔍 Your Job Summary")
    st.dataframe(df.sort_values(["run_id", "step"]), use_container_width=True)

    # Detailed view
    st.markdown("### Detailed Job Info")
    for rid in run_ids:
        with st.expander(f"Run {rid}"):
            dp = dp_jobs.get(rid)
            if dp:
                st.subheader("Data-Prep")
                st.json(dp)
            else:
                st.info("No data-prep info available.")

            ft = ft_jobs.get(rid)
            if ft:
                st.subheader("Fine-Tuning")
                st.json(ft)
            else:
                st.info("No fine-tuning info available.")

        # Active queue overview
    st.markdown("### Active Queue Positions")

    dp_queue_json = raw_dp_queue.get("jobs", {})
    dp_queue_size = raw_dp_queue.get("queue_size", 0)
    ft_queue_json = raw_ft_queue.get("jobs", {})
    ft_queue_size = raw_ft_queue.get("queue_size", 0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data-Prep Queue")
        queued = [
            job
            for job in dp_queue_json.values()
            if job.get("status") in {"queued", "running"}
        ]
        queued.sort(key=lambda j: j.get("position_in_queue", 0))
        st.write(f"**Queue size:** {dp_queue_size}")
        if not queued:
            st.info("No data-prep jobs running or queued.")
        else:
            for job in queued:
                st.json(job)

    with col2:
        st.subheader("Fine-Tuning Queue")
        queued_ft = [
            job
            for job in ft_queue_json.values()
            if job.get("status") in {"queued", "running"}
        ]
        queued_ft.sort(key=lambda j: j.get("position_in_queue", 0))
        st.write(f"**Queue size:** {ft_queue_size}")
        if not queued_ft:
            st.info("No fine-tuning jobs running or queued.")
        else:
            for job in queued_ft:
                st.json(job)

if __name__ == "__main__":
    main() == "__main__":
    main()
