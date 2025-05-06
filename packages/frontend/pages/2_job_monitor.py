"""
Resonare ‑ Job Dashboard

Streamlit page that shows all data‑prep and fine‑tuning jobs plus their
respective queues.  Works even when the backend returns zero jobs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
DATA_PREP_URL: str = os.getenv("DATA_PREP_URL", "http://data-prep:8000")
FINE_TUNE_URL: str = os.getenv("FINE_TUNING_URL", "http://fine-tuning:8000")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _safe_json(url: str, timeout: int = 15) -> Dict[str, Any]:
    """GET <url> and return {} on any error."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json() or {}
    except Exception as exc:  # pylint: disable=broad-except
        st.warning(f"⚠️  Could not reach {url.split('/')[-1]} – {exc}")
        return {}


def get_job_statuses() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (data‑prep jobs, fine‑tune jobs)."""
    return (
        _safe_json(f"{DATA_PREP_URL}/jobs", timeout=20),
        _safe_json(f"{FINE_TUNE_URL}/jobs", timeout=20),
    )


def get_queues() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (data‑prep queue, fine‑tune queue)."""
    return (
        _safe_json(f"{DATA_PREP_URL}/queue"),
        _safe_json(f"{FINE_TUNE_URL}/queue"),
    )


def display_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Side‑by‑side filters that never fail on empty frames."""
    st.markdown("### 🔍 Filter Jobs")

    cols = st.columns(3)
    with cols[0]:
        status_filter = st.multiselect(
            "Status",
            ["queued", "running", "completed", "failed"],
            key="status_filter",
        )
    with cols[1]:
        type_filter = st.multiselect(
            "Job type", ["data-prep", "fine-tuning"], key="type_filter"
        )
    with cols[2]:
        run_id_query = st.text_input("Search run ID", key="runid_search")

    if status_filter:
        df = df[df["status"].isin(status_filter)]
    if type_filter:
        df = df[df["type"].isin(type_filter)]
    if run_id_query:
        df = df[df["run_id"].str.contains(run_id_query, case=False)]

    return df


def display_jobs(dp_jobs: Dict[str, Any], ft_jobs: Dict[str, Any]) -> None:
    """Collapsible JSON panes per job type (works when empty)."""
    st.markdown("## 📋 All Jobs")

    if st.checkbox("Show Data‑Prep Jobs", value=True):
        if not dp_jobs:
            st.info("No data‑prep jobs yet.")
        for run_id, info in sorted(
            dp_jobs.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        ):
            with st.expander(f"[data‑prep] {run_id} — {info.get('status', '‑')}"):
                st.json(info)

    if st.checkbox("Show Fine‑Tuning Jobs", value=True):
        if not ft_jobs:
            st.info("No fine‑tuning jobs yet.")
        for run_id, info in sorted(
            ft_jobs.items(),
            key=lambda kv: kv[1].get("created_at", ""),
            reverse=True,
        ):
            with st.expander(f"[fine‑tune] {run_id} — {info.get('status', '‑')}"):
                st.json(info)


def display_queues(dp_queue: Dict[str, Any], ft_queue: Dict[str, Any]) -> None:
    """Side‑by‑side live queues (empty‑safe)."""
    st.markdown("## 📦 Active Queues")
    q1, q2 = st.columns(2)

    def render(queue: Dict[str, Any], title: str) -> None:
        running = [
            j
            for j in queue.get("jobs", {}).values()
            if j.get("status") in {"queued", "running"}
        ]
        running.sort(key=lambda j: j.get("position_in_queue", 0))

        st.write(f"**Queue size:** {queue.get('queue_size', 0)}")
        if not running:
            st.info(f"No {title.lower()} jobs running or queued.")
        for job in running:
            st.json(job)

    with q1:
        st.subheader("Data‑Prep Queue")
        render(dp_queue, "Data‑Prep")

    with q2:
        st.subheader("Fine‑Tuning Queue")
        render(ft_queue, "Fine‑Tuning")


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #


def main() -> None:
    """Resonare job dashboard (always renders, even if empty)."""
    st.set_page_config(page_title="Resonare – Job Dashboard", layout="wide")
    st.title("📊 Resonare – Job Monitor")
    st.markdown("Track all preprocessing and fine‑tuning jobs in real‑time.")

    dp_jobs, ft_jobs = get_job_statuses()
    dp_queue, ft_queue = get_queues()

    # --- Combined dataframe (guaranteed schema) --------------------------------
    rows: list[dict[str, Any]] = []

    for rid, info in dp_jobs.items():
        rows.append(
            {
                "run_id": rid,
                "type": "data-prep",
                "status": info.get("status", "unknown"),
                "created_at": info.get("created_at", ""),
            }
        )
    for rid, info in ft_jobs.items():
        rows.append(
            {
                "run_id": rid,
                "type": "fine-tuning",
                "status": info.get("status", "unknown"),
                "created_at": info.get("created_at", ""),
            }
        )

    df = pd.DataFrame(rows, columns=["run_id", "type", "status", "created_at"])
    df_filtered = display_filters(df)

    st.markdown("### 📑 Job Table")
    if df_filtered.empty:
        st.info("No matching jobs.")
    else:
        st.dataframe(
            df_filtered.sort_values("created_at", ascending=False),
            use_container_width=True,
        )

    display_jobs(dp_jobs, ft_jobs)
    display_queues(dp_queue, ft_queue)

    st.markdown("### 🔁 Refresh")
    st.caption(
        "Use your browser refresh button to reload. "
        "Automatic refresh has been disabled for stability."
    )


if __name__ == "__main__":
    main()
