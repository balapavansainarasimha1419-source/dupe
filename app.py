import streamlit as st
import os
import time

# Import your own config file that you just built!
import config

# ==========================================
# 1. INTEGRATION SAFETY (The "Don't Crash" Rule)
# ==========================================
# We TRY to import your friend's backend code. 
# Since he hasn't written it yet, this will fail.
# Instead of crashing, we catch the "ImportError" and set a flag.
try:
    from backend.vector_engine import VectorDB
    from backend.parser import extract_text_from_file
    BACKEND_READY = True
except ImportError:
    BACKEND_READY = False


# ==========================================
# 2. STATE MANAGEMENT (Remembering data)
# ==========================================
# Streamlit refreshes the page every time you click a button.
# "session_state" is a memory box. We tell it to remember our scanned files.
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []


# ==========================================
# 3. MAIN PAGE SETUP
# ==========================================
st.set_page_config(page_title="FileSense", page_icon="📂", layout="wide")
st.title("📂 FileSense: AI File Organizer")

# If your friend's code is missing, show a clean warning (not a red error crash)
if not BACKEND_READY:
    st.warning("⚠️ Backend modules not found. UI running in standalone mode (using dummy data).")


# ==========================================
# 4. SIDEBAR LAYOUT (Scanning)
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings & Scanning")
    
    # Text input for the folder path. It defaults to your safe DATA_DIR.
    target_folder = st.text_input("Target Folder Path", value=str(config.DATA_DIR))
    
    if st.button("Scan Directory"):
        # Validate that the folder actually exists
        if os.path.exists(target_folder) and os.path.isdir(target_folder):
            
            # Show a cool progress bar
            progress_text = "Scanning files. Please wait..."
            my_bar = st.progress(0, text=progress_text)
            
            for percent_complete in range(100):
                time.sleep(0.01) # Simulating the time it takes to scan
                my_bar.progress(percent_complete + 1, text=progress_text)
            
            st.success(f"Scanned {target_folder} successfully!")
            
            # TEMPORARY BYPASS: We use dummy data exactly matching your Data Contract
            # When your friend finishes the parser, we will replace this list with his real function.
            st.session_state.scan_results = [
                {'filename': 'machine_learning_notes.pdf', 'filepath': f'{target_folder}/machine_learning_notes.pdf', 'text_content': 'Placeholder text about AI models...', 'metadata': {'size': '2MB'}},
                {'filename': 'project_requirements.docx', 'filepath': f'{target_folder}/project_requirements.docx', 'text_content': 'Placeholder text for requirements...', 'metadata': {'size': '1MB'}}
            ]
        else:
            st.error("Invalid folder path. Please check and try again.")


# ==========================================
# 5. TABS LAYOUT (Search & Cluster)
# ==========================================
# Create the two tabs requested in the prompt
tab1, tab2 = st.tabs(["🔍 Search Files", "🧠 Smart Clusters"])

# --- TAB 1: SEARCH ---
with tab1:
    st.header("Search Your Offline Files")
    search_query = st.text_input("What are you looking for?")
    
    if search_query:
        if len(st.session_state.scan_results) == 0:
            st.info("Please scan a directory first using the sidebar!")
        else:
            st.write(f"Showing results for: **{search_query}**")
            
            # Display results in clean "cards" (containers)
            for file in st.session_state.scan_results:
                with st.container(border=True):
                    st.subheader(file['filename'])
                    st.caption(f"Path: {file['filepath']} | Size: {file['metadata']['size']}")
                    st.write(file['text_content'])

# --- TAB 2: CLUSTERING ---
with tab2:
    st.header("Group Similar Files")
    st.write("Use AI to automatically group your files by topic.")
    
    if st.button("Group Similar Files"):
        if BACKEND_READY:
            # We will connect your friend's vector engine here later
            st.info("Backend connected. Running AI clustering...")
        else:
            # Temporary bypass visual
            st.success("Simulated Clusters generated successfully!")
            
            st.write("### 🤖 Topic: AI & Tech")
            st.write("- machine_learning_notes.pdf")
            
            st.write("### 📝 Topic: Documentation")
            st.write("- project_requirements.docx")     