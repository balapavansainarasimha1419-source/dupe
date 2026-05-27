"""
frontend/search_tab.py

Streamlit render function for Tab 1 — Search Files.

Responsibilities:
    - render search input and filters
    - call vector_engine.search_documents()
    - display results

Non-responsibilities:
    - NO backend logic
    - NO file writing
    - NO state ownership beyond session_state keys it declares
"""

from __future__ import annotations

import datetime
import os
import subprocess
import sys

import streamlit as st

import config


def render(BACKEND_READY: bool) -> None:
    """Render the Search Files tab. Called from app.py inside `with tab_search:`."""

    st.header("Search Your Offline Files")

    search_query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. machine learning project ideas"
    )

    col_type, col_date, col_clear = st.columns([2, 3, 1])

    with col_type:
        available_types = [ext.upper() for ext in config.SUPPORTED_EXTENSIONS]
        type_filter = st.multiselect(
            "File type",
            options=available_types,
            placeholder="All types",
            key="type_filter_key"
        )

    with col_date:
        date_range = st.date_input(
            "Modified between",
            value=[],
            min_value=datetime.date(2000, 1, 1),
            max_value=datetime.date.today(),
            help="Pick a start and end date to filter by last-modified time.",
            key="date_range_key"
        )

    with col_clear:
        st.write("")
        if st.button("✕ Clear", use_container_width=True, key="clear_search_filters"):
            st.session_state.type_filter_key = []
            st.session_state.date_range_key = []
            st.rerun()

    filters_active = bool(type_filter) or (
        isinstance(date_range, (list, tuple)) and len(date_range) == 2
    )

    if not search_query:
        return

    if not BACKEND_READY:
        st.info(f"Dummy search results for: {search_query}")
        return

    with st.spinner("Searching vector space..."):
        raw = st.session_state.vector_db.search_documents(
            query_text=search_query,
            top_k=50,
        )

    if "error" in raw:
        st.info(raw["error"])
        return

    matches = raw.get("matches", [])
    db_meta = st.session_state.vector_db.get_file_metadata()

    if type_filter:
        matches = [
            m for m in matches
            if os.path.splitext(m["filepath"])[1].lower().lstrip(".").upper()
            in {t.lstrip(".") for t in type_filter}
        ]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt, end_dt = date_range
        start_ts = datetime.datetime.combine(start_dt, datetime.time.min).timestamp()
        end_ts   = datetime.datetime.combine(end_dt,   datetime.time.max).timestamp()

        def _mtime(filepath: str) -> float:
            stored = db_meta.get(filepath)
            if stored:
                return float(stored)
            try:
                return os.path.getmtime(filepath)
            except OSError:
                return 0.0

        matches = [
            m for m in matches
            if start_ts <= _mtime(m["filepath"]) <= end_ts
        ]

    count = len(matches)
    if filters_active:
        st.caption(f"{count} result{'s' if count != 1 else ''} after filtering")
    else:
        st.success(f"Found {count} relevant match{'es' if count != 1 else ''}.")

    if not matches:
        st.info("No files matched your query and filters. Try loosening the filters.")
        return

    for match in matches:
        ext = os.path.splitext(match["filepath"])[1].lower()
        icon = {"pdf": "📕", "docx": "📘", "txt": "📄"}.get(ext.lstrip("."), "📄")
        modified_ts = db_meta.get(match["filepath"], 0.0)
        modified_str = (
            datetime.datetime.fromtimestamp(float(modified_ts)).strftime("%d %b %Y")
            if modified_ts else "unknown"
        )

        with st.container(border=True):
            col_info, col_actions = st.columns([5, 2])

            with col_info:
                st.markdown(f"**{icon} {match['filename']}**")
                st.caption(
                    f"`{ext.upper()}` · Modified {modified_str} · "
                    f"Score: {match.get('score', match.get('distance', '—'))}"
                )
                st.write(match["snippet"])

            with col_actions:
                if st.button(
                    "📂 Open",
                    key=f"open_{match['filepath']}",
                    use_container_width=True,
                ):
                    try:
                        if sys.platform == "win32":
                            os.startfile(match["filepath"])
                        elif sys.platform == "darwin":
                            subprocess.Popen(["open", match["filepath"]])
                        else:
                            subprocess.Popen(["xdg-open", match["filepath"]])
                    except Exception as e:
                        st.error(f"Could not open: {e}")

                escaped = match["filepath"].replace("\\", "\\\\").replace("'", "\\'")
                st.components.v1.html(
                    f"""<button onclick="navigator.clipboard.writeText('{escaped}')"
                        style="width:100%;padding:6px 0;border-radius:6px;
                               border:1px solid #ccc;background:#fff;
                               cursor:pointer;font-size:13px;">
                        📋 Copy path
                    </button>""",
                    height=36,
                )
                