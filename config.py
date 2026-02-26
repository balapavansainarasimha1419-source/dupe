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


# ==========================================
# 2. GLOBAL CONSTANTS (The locked-in rules)
# ==========================================
# This is the specific sentence-transformer model the AI will use to understand text.
MODEL_NAME = "all-MiniLM-L6-v2"

# We strictly limit file sizes to prevent the local app from freezing or crashing.
MAX_FILE_SIZE_MB = 10


# ==========================================
# 3. SELF-HEALING SETUP (Crash prevention)
# ==========================================
# This script runs automatically the moment any other file says "import config".
# It checks if our required folders exist. If they don't, it creates them.
# This completely prevents "FileNotFound" errors.

if not DATA_DIR.exists():
    os.makedirs(DATA_DIR)

if not CHROMA_DB_DIR.exists():
    os.makedirs(CHROMA_DB_DIR)

# A simple print statement so we know the setup worked when we start the app.
print("✅ Configuration loaded. Data directories ready.")