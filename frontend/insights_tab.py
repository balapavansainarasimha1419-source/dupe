"""
frontend/insights_tab.py

Streamlit render function for Tab 5 — Insights.

Responsibilities:
    - display file type distribution chart
    - display summary metrics
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st


def render(BACKEND_READY: bool) -> None:
    """Render the Insights tab. Called from app.py inside `with tab_insights:`."""

    st.header("📊 Database Insights")

    if not BACKEND_READY or not st.session_state.vector_db:
        st.info("Backend not available.")
        return

    db_files = st.session_state.vector_db.get_file_metadata()

    if not db_files:
        st.info("No data to visualize. Scan a folder first!")
        return

    file_types: dict[str, int] = {}
    for fp in db_files.keys():
        ext = os.path.splitext(fp)[1].lower().replace(".", "").upper() or "UNKNOWN"
        file_types[ext] = file_types.get(ext, 0) + 1

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Distribution by File Type")
        chart_data = pd.DataFrame(
            {"Count": list(file_types.values())},
            index=list(file_types.keys()),
        )
        st.bar_chart(chart_data, color="#1E88E5")

    with col2:
        st.subheader("Summary")
        st.metric("Total Files", len(db_files))
        st.write(f"Your primary format is **{max(file_types, key=file_types.get)}**.")