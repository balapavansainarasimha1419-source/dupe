"""
app.py — FileSense V1.5

Thin orchestration layer. Owns:
    - page config
    - global session state initialisation
    - sidebar (scanning + settings)
    - tab layout

Does NOT own:
    - tab rendering logic  (frontend/*.py)
    - backend logic        (backend/*.py)
"""

import streamlit as st
import os
import time

# ==========================================
# 0. PAGE CONFIG — must be first Streamlit call
# ==========================================
st.set_page_config(page_title="FileSense", page_icon="📂", layout="wide")

# ==========================================
# 1. BACKEND IMPORTS
# ==========================================
try:
    import config
    from backend.vector_engine import VectorDB
    from backend.parser import extract_text_from_file
    from backend.undo_stack import UndoStack
    BACKEND_READY = True
except ImportError as e:
    BACKEND_READY = False
    print(f"Import Error: {e}")

# ==========================================
# 2. GLOBAL SESSION STATE
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []

if 'vector_db' not in st.session_state:
    st.session_state.vector_db = VectorDB() if BACKEND_READY else None

if 'undo_stack' not in st.session_state:
    st.session_state.undo_stack = UndoStack() if BACKEND_READY else None

if 'pending_deletes' not in st.session_state:
    st.session_state.pending_deletes = []

if 'cluster_names' not in st.session_state:
    st.session_state.cluster_names = {}

if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = {}

# Tab 3 editor state is initialised by editor_tab.init_state()
# Called below after imports are confirmed ready.
if BACKEND_READY:
    from frontend.editor_tab import init_state as _init_editor_state
    _init_editor_state()

# ==========================================
# 3. HEADER
# ==========================================
st.title("📂 FileSense: AI File Organizer")

if not BACKEND_READY:
    st.warning("⚠️ Backend modules not found. UI running in standalone mode.")
else:
    st.success("✅ Backend fully connected. Local AI Engine is active.")

# ==========================================
# 4. SIDEBAR — scanning & settings
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Scanning")

    default_path = str(config.DATA_DIR) if BACKEND_READY else ""
    target_folder = st.text_input(
        "Target Folder Path",
        value=default_path,
        key="unique_folder_input"
    ).strip()

    preserve_structure = st.checkbox("📁 Keep Folder Structure (Skip AI Clustering)")

    if st.button("🚀 Fast Scan Directory"):
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            valid_exts = config.SUPPORTED_EXTENSIONS
            current_filepaths = []
            total_files_seen  = 0

            for root, dirs, files in os.walk(target_folder):
                total_files_seen += len(files)
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_exts:
                        current_filepaths.append(os.path.join(root, file))

            st.warning(f"🔍 DEBUG: Found {total_files_seen} total files across all sub-folders.")
            st.info(f"🔍 DEBUG: {len(current_filepaths)} files matched supported formats.")

            if BACKEND_READY:
                st.info("Syncing entire directory tree with AI memory...")

                db_files           = st.session_state.vector_db.get_file_metadata()
                memorized_filepaths = list(db_files.keys())

                ghosts_removed = 0
                for ghost_path in memorized_filepaths:
                    if ghost_path not in current_filepaths:
                        st.session_state.vector_db.remove_file(ghost_path)
                        ghosts_removed += 1

                files_to_process = []
                for filepath in current_filepaths:
                    if filepath not in memorized_filepaths:
                        files_to_process.append(filepath)
                    else:
                        if os.path.getmtime(filepath) > db_files.get(filepath, 0.0):
                            files_to_process.append(filepath)

                if not files_to_process and ghosts_removed == 0:
                    st.success("Everything is already up to date! No new changes detected.")
                else:
                    my_bar          = st.progress(0, text="Initializing AI analysis...")
                    processed_count = 0
                    total_to_process = len(files_to_process)

                    for idx, filepath in enumerate(files_to_process):
                        parsed_data = extract_text_from_file(filepath)

                        if not parsed_data.get('error'):
                            st.session_state.vector_db.add_file(
                                filename=parsed_data['filename'],
                                filepath=parsed_data['filepath'],
                                text=parsed_data['text_content'],
                                mtime=os.path.getmtime(filepath),
                                preserve_structure=preserve_structure,
                                parent_folder=os.path.basename(os.path.dirname(filepath)),
                            )
                            processed_count += 1

                        my_bar.progress(
                            int(((idx + 1) / total_to_process) * 100),
                            text=f"Vectorizing: {os.path.basename(filepath)}",
                        )

                    st.success(
                        f"✅ Sync complete! Scanned {processed_count} files "
                        f"and removed {ghosts_removed} deleted files."
                    )
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("AI Backend failed to load. Check terminal.")
        else:
            st.error(f"❌ Path Error: The folder '{target_folder}' does not exist.")

    st.divider()
    st.subheader("Danger Zone")
    if st.button("🚨 Wipe AI Memory", type="primary", use_container_width=True):
        if BACKEND_READY:
            with st.spinner("Erasing knowledge..."):
                st.session_state.vector_db.clear_database()
                st.session_state.undo_stack.clear()
                st.session_state.pending_deletes.clear()
                st.session_state.scan_results    = []
                st.session_state.cluster_results = {}
            st.success("✅ AI Memory wiped!")
            time.sleep(1)
            st.rerun()

# ==========================================
# 5. TAB LAYOUT — thin router only
# ==========================================
tab_search, tab_cluster, tab_editor, tab_manage, tab_insights = st.tabs([
    "🔍 Search Files",
    "🧠 Smart Clusters",
    "🤖 AI Editor",
    "🗄️ Manage Files",
    "📊 Insights",
])

from frontend import search_tab, cluster_tab, editor_tab, manage_tab, insights_tab

with tab_search:
    search_tab.render(BACKEND_READY)

with tab_cluster:
    cluster_tab.render(BACKEND_READY, target_folder)

with tab_editor:
    editor_tab.render(BACKEND_READY)

with tab_manage:
    manage_tab.render(BACKEND_READY)

with tab_insights:
    insights_tab.render(BACKEND_READY)
    