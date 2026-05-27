"""
frontend/manage_tab.py

Streamlit render function for Tab 4 — Manage Files.

Responsibilities:
    - display all indexed files
    - trigger individual file removal from vector DB
"""

from __future__ import annotations

import os
import time

import streamlit as st


def render(BACKEND_READY: bool) -> None:
    """Render the Manage Files tab. Called from app.py inside `with tab_manage:`."""

    st.header("Database File Management")

    if not BACKEND_READY or not st.session_state.vector_db:
        st.info("Backend not available.")
        return

    db_files = st.session_state.vector_db.get_file_metadata()
    active_filepaths = [
        fp for fp in db_files.keys()
        if fp not in st.session_state.pending_deletes
    ]

    if not active_filepaths:
        st.info("No active files found in the database.")
        return

    for filepath in active_filepaths:
        filename = os.path.basename(filepath)
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**📄 {filename}**")
            st.caption(filepath)
        with col2:
            if st.button("Delete", key=f"del_{filepath}"):
                st.session_state.vector_db.remove_file(filepath)
                st.toast(f"Removed '{filename}' from AI memory.")
                time.sleep(0.5)
                st.rerun()