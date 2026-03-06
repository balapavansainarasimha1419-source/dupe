import streamlit as st
import os
import time
import config

# ==========================================
# 0. PAGE CONFIG (Must be the absolute first command!)
# ==========================================
st.set_page_config(page_title="FileSense", page_icon="📂", layout="wide")

# ==========================================
# 1. INTEGRATION SAFETY & BACKEND LOADING
# ==========================================
try:
    from backend.vector_engine import VectorDB
    from backend.parser import extract_text_from_file
    BACKEND_READY = True
except ImportError:
    BACKEND_READY = False

# ==========================================
# 2. STATE MANAGEMENT 
# ==========================================
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
    
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am your local AI. How can I help you analyze or summarize your files today?"}
    ]

# Initialize the AI Database exactly once to save memory
if 'vector_db' not in st.session_state:
    if BACKEND_READY:
        st.session_state.vector_db = VectorDB()
    else:
        st.session_state.vector_db = None

# ==========================================
# 3. MAIN PAGE SETUP
# ==========================================
st.title("📂 FileSense: AI File Organizer")

if not BACKEND_READY:
    st.warning("⚠️ Backend modules not found. UI running in standalone mode (using dummy data).")
else:
    st.success("✅ Backend fully connected. Local AI Engine is active.")

# ==========================================
# 4. SIDEBAR LAYOUT (Smart Sync Logic!)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Scanning")
    target_folder = st.text_input("Target Folder Path", value=str(config.DATA_DIR))
    
    if st.button("Scan Directory"):
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            if BACKEND_READY:
                st.info("Syncing folder with AI memory...")
                
                valid_extensions = ['.pdf', '.docx', '.txt']
                files_to_scan = [f for f in os.listdir(target_folder) if os.path.splitext(f)[1].lower() in valid_extensions]
                current_filepaths = [os.path.join(target_folder, f) for f in files_to_scan]
                
                # 1. Roll Call: Ask what the AI remembers AND when it was last edited
                db_files = st.session_state.vector_db.get_file_metadata()
                memorized_filepaths = list(db_files.keys())
                
                # 2. Forget the Ghosts
                ghosts_removed = 0
                for ghost_path in memorized_filepaths:
                    if ghost_path not in current_filepaths:
                        st.session_state.vector_db.remove_file(ghost_path)
                        ghosts_removed += 1
                        
                # 3. Find New OR Modified Files
                files_to_process = []
                for filepath in current_filepaths:
                    if filepath not in memorized_filepaths:
                        files_to_process.append(filepath) # Brand new file
                    else:
                        # Exists, but did the user edit it? Check the timestamp!
                        current_mtime = os.path.getmtime(filepath)
                        saved_mtime = db_files.get(filepath, 0.0)
                        
                        if current_mtime > saved_mtime:
                            files_to_process.append(filepath) # The file was modified!
                
                if not files_to_process and ghosts_removed == 0:
                    st.success("Everything is already up to date! No new or modified files to scan.")
                else:
                    if files_to_process:
                        progress_text = "Parsing new and updated files..."
                        my_bar = st.progress(0, text=progress_text)
                        total_files = len(files_to_process)
                        
                        st.session_state.scan_results = [] 
                        
                        for idx, filepath in enumerate(files_to_process):
                            filename = os.path.basename(filepath)
                            parsed_data = extract_text_from_file(filepath)
                            
                            if not parsed_data.get('error'):
                                st.session_state.vector_db.add_file(
                                    filename=parsed_data['filename'],
                                    filepath=parsed_data['filepath'],
                                    text=parsed_data['text_content']
                                )
                                st.session_state.scan_results.append(parsed_data)
                                
                            percent_complete = int(((idx + 1) / total_files) * 100)
                            my_bar.progress(percent_complete, text=f"Processed {filename}...")
                    
                    st.success(f"Sync complete! Removed {ghosts_removed} deleted files and updated/scanned {len(files_to_process)} files.")
            else:
                # Fallback dummy data if backend fails to load
                time.sleep(1)
                st.success("Scanned successfully (Dummy Data mode).")
                st.session_state.scan_results = [{'filename': 'dummy.pdf', 'filepath': 'path', 'text_content': 'dummy', 'metadata': {'size': '1MB'}}]
        else:
            st.error("Invalid folder path. Please check and try again.")

# ==========================================
# 5. TABS LAYOUT 
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 Search Files", "🧠 Smart Clusters", "🤖 AI Editor"])

# --- TAB 1: SEARCH ---
with tab1:
    st.header("Search Your Offline Files")
    search_query = st.text_input("What are you looking for?")
    
    if search_query:
        if not BACKEND_READY:
            st.info(f"Dummy search results for: {search_query}")
        else:
            # REAL AI VECTOR SEARCH
            results = st.session_state.vector_db.search(search_query)
            
            if not results:
                st.info("No matching documents found.")
            else:
                st.write(f"Top results for: **{search_query}**")
                for file in results:
                    with st.container(border=True):
                        st.subheader(file['filename'])
                        st.caption(f"Path: {file['filepath']}")
                        # Show a preview snippet of the text
                        st.write(file['text_content'][:300] + "...")

# --- TAB 2: CLUSTERING ---
with tab2:
    st.header("Group Similar Files")
    st.write("Use AI to automatically group your files by topic.")
    
    if st.button("Group Similar Files"):
        if BACKEND_READY:
            # REAL K-MEANS CLUSTERING
            cluster_results = st.session_state.vector_db.cluster_files()
            
            if 'error' in cluster_results:
                st.error(cluster_results['error'])
            elif 'warning' in cluster_results:
                st.warning(cluster_results['warning'])
            else:
                st.success("Clusters generated successfully!")
                for cluster_id, files in cluster_results.items():
                    st.write(f"### 🤖 Cluster {cluster_id + 1}")
                    for f in files:
                        st.write(f"- {f}")
        else:
            st.success("Simulated Clusters generated successfully!")

# --- TAB 3: AI EDITOR ---
with tab3:
    st.header("Offline AI Editor")
    st.caption("Strictly routes to local Ollama engine.")
    
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    if prompt := st.chat_input("Ask the AI to analyze, summarize, or draft new text..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            dummy_reply = f"I am running locally! You asked: '{prompt}'. (Ollama LLM connection coming in Phase 2!)"
            st.write(dummy_reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": dummy_reply})