"""
frontend/cluster_tab.py

Streamlit render function for Tab 2 — Smart Clusters.

Responsibilities:
    - render cluster controls and results
    - call vector_engine.cluster_files()
    - handle push-to-folders file operations

Non-responsibilities:
    - NO backend logic
    - NO embedding or clustering math
    - NO file writing via file_writer (cluster copy/move is intentionally
      direct shutil — it is not AI-generated content and needs no validators)
"""

from __future__ import annotations

import os
import shutil
import time

import streamlit as st


def render(BACKEND_READY: bool, target_folder: str) -> None:
    """Render the Smart Clusters tab. Called from app.py inside `with tab_cluster:`."""

    st.header("Group Similar Files")
    st.write("Use AI to automatically group your files by topic.")

    col_run, col_reset = st.columns([3, 1])

    with col_run:
        if st.button("🧠 Group Similar Files", use_container_width=True):
            if BACKEND_READY:
                raw = st.session_state.vector_db.cluster_files()
                if "error" in raw:
                    st.error(raw["error"])
                elif "warning" in raw:
                    st.warning(raw["warning"])
                else:
                    st.session_state.cluster_results = raw
                    st.session_state.cluster_names = {}
            else:
                st.warning("Backend not available.")

    with col_reset:
        if st.button("✕ Clear", use_container_width=True, key="clear_clusters"):
            st.session_state.cluster_results = {}
            st.session_state.cluster_names   = {}
            st.rerun()

    if not st.session_state.get("cluster_results"):
        return

    cluster_results = st.session_state.cluster_results

    def _sort_key(item):
        label = item[0]
        if label.startswith("Cluster "):
            try:
                return (0, int(label.split(" ")[1]))
            except ValueError:
                return (0, float("inf"))
        return (1, label)

    sorted_clusters = sorted(cluster_results.items(), key=_sort_key)
    total_files     = sum(len(f) for _, f in sorted_clusters)

    st.success(
        f"Found **{len(sorted_clusters)}** group(s) across **{total_files}** file(s). "
        "Edit any name below before pushing to folders."
    )

    for original_label, files in sorted_clusters:
        is_noise = original_label == "Uncategorized / Noise"

        with st.container(border=True):
            name_col, badge_col = st.columns([5, 1])

            with name_col:
                icon        = "🗂️" if is_noise else "📁"
                current_val = st.session_state.cluster_names.get(
                    original_label, original_label
                )
                new_name = st.text_input(
                    "Cluster name",
                    value            = current_val,
                    key              = f"cname_{original_label}",
                    label_visibility = "collapsed",
                    placeholder      = "Type a folder name…",
                    disabled         = is_noise,
                    help             = (
                        "This name becomes the folder name when you push to disk."
                        if not is_noise
                        else "Uncategorized files are always placed in a fixed noise folder."
                    ),
                )
                st.session_state.cluster_names[original_label] = (
                    original_label if is_noise else (new_name.strip() or original_label)
                )

            with badge_col:
                st.metric(label="files", value=len(files))

            with st.expander(
                f"{icon}  {len(files)} file{'s' if len(files) != 1 else ''}",
                expanded=not is_noise,
            ):
                for f in sorted(files, key=lambda x: x["filename"]):
                    st.caption(f"📄 **{f['filename']}** — `{f['filepath']}`")

    st.divider()
    st.subheader("📁 Push Clusters to Folders")

    with st.expander("Preview folder names", expanded=False):
        for original_label, files in sorted_clusters:
            resolved = st.session_state.cluster_names.get(original_label, original_label)
            safe     = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_"
                for c in resolved
            ).strip()
            st.caption(f"`{safe}/`  ← {len(files)} file(s)")

    col_path, col_mode = st.columns([3, 1])
    with col_path:
        output_base = st.text_input(
            "Base output folder",
            value=target_folder,
            help="Cluster subfolders will be created inside this directory.",
        )
    with col_mode:
        operation = st.radio(
            "Operation",
            ["Copy", "Move"],
            captions=["Keeps originals", "Removes originals"],
        )

    if not st.button("🚀 Create Folders & Organize", type="primary"):
        return

    if not output_base or not os.path.isdir(output_base):
        st.error("❌ Output folder does not exist. Please enter a valid path.")
        return

    total_ok   = 0
    total_skip = 0
    errors     = []

    all_tasks = [
        (label, f)
        for label, files in sorted_clusters
        for f in files
    ]
    progress = st.progress(0, text="Organising files…")

    for idx, (original_label, f) in enumerate(all_tasks):
        src      = f["filepath"]
        filename = f["filename"]

        resolved_name = st.session_state.cluster_names.get(original_label, original_label)
        safe_label = "".join(
            c if c.isalnum() or c in (" ", "-", "_") else "_"
            for c in resolved_name
        ).strip()

        dest_dir  = os.path.join(output_base, safe_label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        if (
            os.path.exists(dest_path)
            and os.path.abspath(src) != os.path.abspath(dest_path)
        ):
            base, ext = os.path.splitext(filename)
            counter   = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir, f"{base} ({counter}){ext}")
                counter  += 1

        try:
            if not os.path.exists(src):
                errors.append(f"Source missing: {src}")
                total_skip += 1
            elif os.path.abspath(src) == os.path.abspath(dest_path):
                total_skip += 1
            else:
                (shutil.copy2 if operation == "Copy" else shutil.move)(src, dest_path)
                total_ok += 1
        except Exception as e:
            errors.append(f"{filename}: {e}")

        progress.progress(
            int(((idx + 1) / len(all_tasks)) * 100),
            text=f"{operation}ing: {filename}",
        )

    st.success(
        f"✅ Done — {total_ok} file(s) {operation.lower()}d, "
        f"{total_skip} skipped."
    )
    if errors:
        with st.expander(f"⚠️ {len(errors)} error(s)"):
            for err in errors:
                st.caption(err)

    if operation == "Move" and total_ok > 0:
        st.info("Files moved — re-scanning to update AI memory…")
        st.session_state.cluster_results = {}
        st.session_state.cluster_names   = {}
        time.sleep(1)
        st.rerun()