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

# FIX — OFFLINE GUARANTEE: Local folder where the embedding model lives.
# This is the ONLY place vector_engine.py is allowed to load models from.
# The model is downloaded here once by setup_models.py, then never fetched from
# the internet again. This folder travels with the project — it is not a system cache.
MODEL_CACHE_DIR = BASE_DIR / "models"


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
OLLAMA_MODEL = "gemma3:4b"

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

# OFFLINE GUARANTEE: Create the local model cache folder if it doesn't exist yet.
# setup_models.py will populate it. vector_engine.py will refuse to load from anywhere else.
if not MODEL_CACHE_DIR.exists():
    os.makedirs(MODEL_CACHE_DIR)

# Confirmation message on startup.
print("✅ Configuration loaded. Data directories ready.")


# ============================================================
# Filler Word Cleaning
# ============================================================
ENABLE_FILLER_CLEANING = True
FILLER_FREQUENCY_THRESHOLD = 0.03
FILLER_MIN_WORD_LENGTH = 3
MIN_WORDS_FOR_CLEANING = 50


# ================================================================
# AI EDITOR SETTINGS
# ================================================================

# Maximum estimated tokens before rejecting a file as too large
# for safe deterministic full-file editing.
#
# Hardware: HP Victus, RTX 2050 (4GB VRAM)
#   - Gemma 3 4B weights  : ~2.5GB VRAM
#   - Remaining for KV    : ~1.5GB → safe num_ctx = 8192
#   - Prompt overhead     : ~500 tokens
#
# IMPORTANT:
# VRAM capacity is NOT the same as reliable output discipline.
#
# Gemma 3 4B can technically fit more than 3500 tokens,
# but deterministic full-file regeneration becomes unstable
# beyond this range. At larger sizes it may:
#   - truncate output
#   - drop closing XML tags
#   - silently omit sections of the file
#
# Large files should use section-level editing instead
# of full-file regeneration.
MAX_FILE_TOKENS = 3500

# Context window passed to Ollama during edit generation.
#
# Tuned specifically for RTX 2050 4GB VRAM headroom.
# Increasing this without stronger hardware may cause:
#   - VRAM exhaustion
#   - generation instability
#   - slower inference
OLLAMA_NUM_CTX = 8192

# Maximum undo snapshots stored per editing session.
# Oldest snapshot is dropped silently once the cap is reached.
UNDO_STACK_LIMIT = 10

# Ollama sampling settings for deterministic editing.
#
# Low temperature forces stronger instruction-following
# instead of improvisation — critical for:
#   - XML output discipline
#   - stable formatting
#   - reproducible edits
OLLAMA_TEMPERATURE = 0.1
OLLAMA_TOP_P = 0.9

# Maximum allowed output size relative to original file size.
#
# Example:
#   original   = 1000 chars
#   max output = 3000 chars
#
# Prevents runaway hallucinated file expansion.
MAX_EDIT_SIZE_MULTIPLIER = 3.0

# Timeout in seconds for AI edit generation.
#
# Separate from INFERENCE_TIMEOUT intentionally.
# Editing and chat are different workloads and should
# be independently tunable.
OLLAMA_EDIT_TIMEOUT = 120

OLLAMA_CHAT_TEMPERATURE = 0.2
OLLAMA_EXPLAIN_TEMPERATURE = 0.3

# ================================================================
# PROTECTED DIRECTORIES
# ================================================================

# Directories the AI editor must NEVER modify.
#
# Checked at:
#   1. context-build time (early rejection before Ollama is called)
#   2. file-write time   (final safety guard before disk write)
#
# Both checks are intentional — do not remove either layer.
PROTECTED_PATHS = [
    ".git",
    "__pycache__",
    "venv",
    "node_modules",
    "data/chroma_db",
    "models",
    ".cache",
]
