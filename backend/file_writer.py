"""
Atomic disk-write trust boundary for FileSense V1.5.

This module is the ONLY subsystem allowed to mutate filesystem state.

Core philosophy:
    filesystem = source of truth
    file_writer = mutation authority

Responsibilities:
    - write files atomically
    - prevent partial writes
    - preserve UTF-8 encoding
    - reject protected paths independently
    - fail closed on all disk errors
    - support rollback-safe replacement

Non-responsibilities (enforced, not aspirational):
    - NO Ollama calls             (ollama_bridge.py)
    - NO syntax validation        (validators.py)
    - NO diff generation          (Tab 3 UI / difflib)
    - NO undo stack management    (app.py Tab 3)
    - NO Streamlit / PySide6
    - NO token budgeting          (token_budget.py)

Run from project root. No sys.path manipulation.

---

WRITE PIPELINE CONTRACT (caller's responsibility):

    AI generation
        ↓
    XML extraction
        ↓
    validators.py
        ↓
    diff preview
        ↓
    push original to undo stack
        ↓
    atomic_write()

Rejected or failed writes MUST NOT enter undo history.
This module does not enforce that rule — the caller does.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from backend.token_budget import _is_protected


# ============================================================================
# PUBLIC API
# ============================================================================

def atomic_write(filepath: str, content: str) -> dict:
    """
    Atomically replace a file's content with validated edited content.

    This is the final critical trust boundary in the FileSense pipeline.
    It is the ONLY function in the codebase permitted to write to disk.

    Write sequence:
        1. validate filepath
        2. validate content type
        3. reject protected paths (independent of token_budget.py check)
        4. verify target directory exists — never auto-create
        5. pre-encode content as UTF-8 — fail before touching disk
        6. write to temp file in same directory as target
        7. flush() Python-level buffers
        8. os.fsync() OS-level buffers
        9. close temp file (required for Windows os.replace() safety)
        10. os.replace() atomically swaps temp → target
        11. null out temp_path to prevent finally-block false cleanup
        12. on any failure: leave original file untouched, clean up temp

    Parameters
    ----------
    filepath : Absolute path to the file being replaced.
    content  : Fully validated UTF-8 string to write.
               Caller MUST pass validators.validate_edit() before calling.

    Returns
    -------
    Success:
    {
        "success": True,
        "error":   None,
    }

    Failure:
    {
        "success": False,
        "error":   str,   # Human-readable. Show directly to user.
    }

    Caller contract
    ---------------
    "success": False → show result["error"] to user. Original file unchanged.
    "success": True  → file has been atomically replaced on disk.

    WHY os.replace() AND NOT shutil.move() OR os.rename()
    ------------------------------------------------------
    os.replace() is atomic on Windows, Linux, and macOS.
    shutil.move() degrades to copy+delete across filesystems — not atomic.
    os.rename() raises on Windows if the target already exists.
    os.replace() is the only cross-platform atomic option.

    Temp file is placed in the same directory as the target deliberately:
    os.replace() atomicity is guaranteed only within a single filesystem.
    A temp file on a different mount point silently degrades to copy+delete.

    WHY PROTECTED PATH CHECK IS DUPLICATED HERE
    --------------------------------------------
    token_budget.py also calls _is_protected() at context-build time.
    Both checks are intentional. Trust boundaries require redundant
    enforcement. If the UI check regresses, this layer still holds.
    Safety > permissiveness.
    """

    # ----------------------------------------------------------------------
    # Guard 1 — Filepath presence
    # ----------------------------------------------------------------------

    if not filepath or not str(filepath).strip():
        return {
            "success": False,
            "error": "Invalid filepath: empty or whitespace.",
        }

    # ----------------------------------------------------------------------
    # Guard 2 — Content type
    # ----------------------------------------------------------------------

    if not isinstance(content, str):
        return {
            "success": False,
            "error": "Content must be a str. Received: {}.".format(
                type(content).__name__
            ),
        }

    # ----------------------------------------------------------------------
    # Guard 3 — Protected path enforcement
    #
    # Intentionally duplicates the check in token_budget.py.
    # This layer must fail closed independently.
    # ----------------------------------------------------------------------

    if _is_protected(filepath):
        return {
            "success": False,
            "error": (
                "Write rejected: '{}' is inside a protected directory."
                .format(os.path.basename(filepath))
            ),
        }

    target_path = Path(filepath).resolve()
    parent_dir = target_path.parent

    # ----------------------------------------------------------------------
    # Guard 4 — Target directory must exist
    #
    # Do NOT auto-create. Silent directory creation can mask hallucinated
    # or invalid upstream paths from the AI pipeline.
    # ----------------------------------------------------------------------

    if not parent_dir.exists():
        return {
            "success": False,
            "error": (
                "Write rejected: target directory does not exist:\n{}"
                .format(parent_dir)
            ),
        }

    # ----------------------------------------------------------------------
    # Guard 5 — UTF-8 pre-encode validation
    #
    # Fail before any filesystem operation.
    # validators.py should have caught this already, but trust boundaries
    # are redundant by design.
    # ----------------------------------------------------------------------

    try:
        content.encode("utf-8")
    except UnicodeEncodeError as e:
        return {
            "success": False,
            "error": "UTF-8 encoding failed: {}".format(str(e)),
        }

    # -------------------------------------------------------------------------
    # Atomic write sequence
    # -------------------------------------------------------------------------

    temp_path: str | None = None

    try:
        # ------------------------------------------------------------------
        # Step 1 — Write to temp file in same directory as target.
        #
        # Same-directory placement is intentional.
        # os.replace() atomicity requires source and target to be on the
        # same filesystem. A different directory risks a different mount
        # point, which silently degrades os.replace() to copy+delete.
        # ------------------------------------------------------------------

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",           # Suppress platform newline normalization.
            delete=False,         # We manage deletion manually.
            dir=str(parent_dir),
            prefix=".filesense_tmp_",
        ) as temp_file:

            temp_path = temp_file.name

            # Step 2 — Write content
            temp_file.write(content)

            # Step 3 — Flush Python-level write buffers
            temp_file.flush()

            # Step 4 — Flush OS-level write buffers to physical storage
            os.fsync(temp_file.fileno())

        # ------------------------------------------------------------------
        # Context manager exit closes the file handle here.
        #
        # CRITICAL on Windows:
        # os.replace() will raise PermissionError if the source file handle
        # is still open. The with-block close must complete before replace.
        # ------------------------------------------------------------------

        # Step 5 — Atomic replacement
        os.replace(temp_path, str(target_path))

        # Step 6 — Null out temp_path.
        #
        # os.replace() renamed temp_path → target_path.
        # The path temp_path no longer exists on disk.
        # Setting to None prevents the finally block from attempting a
        # redundant os.path.exists() check on a now-stale path string.
        # Without this, the finally block relies on the filesystem side
        # effect to protect against a double-unlink — correct but fragile.
        temp_path = None

        return {
            "success": True,
            "error": None,
        }

    except PermissionError:
        return {
            "success": False,
            "error": (
                "Permission denied. The file may be read-only or "
                "locked by another application."
            ),
        }

    except OSError as e:
        return {
            "success": False,
            "error": "Filesystem error during atomic write: {}".format(str(e)),
        }

    except Exception as e:
        # Broad catch is intentional at trust-boundary infrastructure level.
        # This layer must never propagate exceptions to the UI.
        return {
            "success": False,
            "error": "Unexpected atomic write failure: {}".format(str(e)),
        }

    finally:
        # ------------------------------------------------------------------
        # Fail-closed temp file cleanup.
        #
        # Runs on every exit path — success, failure, and exception.
        # On success: temp_path is None (nulled above). Block is a no-op.
        # On failure: temp_path holds the stale temp file path. Delete it.
        #
        # OSError during cleanup is swallowed intentionally.
        # This block must NEVER mask or replace the primary exception.
        # ------------------------------------------------------------------

        if temp_path is not None:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError:
                pass