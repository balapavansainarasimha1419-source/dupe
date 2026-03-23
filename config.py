import os
from pathlib import Path

# ==========================================
# 1. PATH MANAGEMENT (Where things live)
# ==========================================
# BASE_DIR automatically finds the exact folder where this config.py file is located.
# This ensures the code works on your laptop, my laptop, or anywhere else.
BASE_DIR = Path(__file__).resolve().parent

# We define a "data" folder to store everything the app creates.
DATA_DIR = BASE_DIR / "data"

# Inside the "data" folder, we define a specific folder for our ChromaDB (Vector Database).
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# FIX 1 (Phase 2): Temporary folder strictly for intermediate file conversions (e.g. pdf2docx).
# Files here are never shown to the user and are cleaned up after processing.
TEMP_DIR = BASE_DIR / "temp"


# ==========================================
# 2. GLOBAL CONSTANTS (The locked-in rules)
# ==========================================
# The sentence-transformer model the AI uses to understand text.
# vector_engine.py reads this — change the model name here only, nowhere else.
MODEL_NAME = "all-MiniLM-L6-v2"

# FIX 2: Updated from 10 → 100 to match the actual enforcement in parser.py.
# Previously config said 10 MB but parser was silently enforcing 100 MB — they disagreed.
# parser.py reads this constant directly so both are now in sync.
MAX_FILE_SIZE_MB = 100

# FIX 3: Added SUPPORTED_EXTENSIONS — was missing from config entirely.
# Previously app.py and parser.py each hardcoded ['.pdf', '.docx', '.txt'] independently.
# Now there is ONE place to update when new file types are added (e.g. media files to-do).
# Both app.py and parser.py must import and use this list instead of their own hardcoded versions.
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

# ==========================================
# 3. PHASE 2 CONSTANTS (Local LLM — Ollama)
# ==========================================
# FIX 4 (Phase 2): These constants are required by the Phase 2 protocol spec.
# backend/editor.py will read these. Do not hardcode them in editor.py.

# The local Ollama model to use for the AI Editor feature.
OLLAMA_MODEL = "llama3"

# Hard cap on words sent to the local model per request.
# Prevents CPU crashes on standard hardware by limiting context window size.
MAX_CONTEXT_WORDS = 1000

# Seconds before a local inference request is considered timed out and aborted.
INFERENCE_TIMEOUT = 120


# ==========================================
# 4. SELF-HEALING SETUP (Crash prevention)
# ==========================================
# This script runs automatically the moment any other file says "import config".
# It checks if our required folders exist. If they don't, it creates them.
# This completely prevents "FileNotFound" errors on fresh installs.

if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)

if not CHROMA_DB_DIR.exists():
    os.makedirs(CHROMA_DB_DIR)

# FIX 5 (Phase 2): Added TEMP_DIR to the self-healing setup.
# Required by the Phase 2 protocol spec for temporary file conversion storage.
if not TEMP_DIR.exists():
    os.makedirs(TEMP_DIR)

# Confirmation message on startup.
print("✅ Configuration loaded. Data directories ready.")
