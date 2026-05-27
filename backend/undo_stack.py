"""
backend/undo_stack.py

Self-contained undo history manager for FileSense V1.5.

Core philosophy:
    The UI triggers actions.
    The backend owns state.

Responsibilities:
    - store file content snapshots with metadata
    - enforce the undo stack size cap
    - expose a clean push/pop/peek API
    - be serialization-friendly for session state storage
    - be portable across Streamlit and PySide6

Non-responsibilities (enforced, not aspirational):
    - NO file writing          (file_writer.py)
    - NO diff generation       (diff_preview.py)
    - NO validators            (validators.py)
    - NO Ollama calls          (ollama_bridge.py)
    - NO Streamlit / PySide6

Portability rule:
    This class must contain zero UI framework imports.
    Streamlit stores an instance of this class in st.session_state.
    PySide6 will store it in application state.
    Neither framework should need to know how this class works internally.

Run from project root. No sys.path manipulation.
"""

from __future__ import annotations

import datetime

import config


# ============================================================================
# UNDO ENTRY STRUCTURE
# ============================================================================
#
# Each snapshot is a plain dict — serializable, framework-agnostic.
#
# FROZEN CONTRACT:
# {
#     "filepath":    str,          # absolute path of the file that was edited
#     "content":     str,          # full file content BEFORE the edit
#     "timestamp":   str,          # ISO 8601 UTC timestamp of the snapshot
#     "instruction": str | None,   # the edit instruction that caused this change
# }
#
# "instruction" is optional in V1.5 but present for future diff-summary features.
# Do not remove it — downstream tooling will use it in V2.


def _make_entry(
    filepath: str,
    content: str,
    instruction: str | None = None,
) -> dict:
    """
    Build a single undo snapshot entry.

    Internal factory. Not part of the public API.
    """
    return {
        "filepath":    filepath,
        "content":     content,
        "timestamp":   datetime.datetime.utcnow().isoformat() + "Z",
        "instruction": instruction or "",
    }


# ============================================================================
# PUBLIC CLASS
# ============================================================================

class UndoStack:
    """
    Bounded undo history for the FileSense AI Editor.

    Stores full file content snapshots before each accepted edit.
    The oldest snapshot is silently evicted when the cap is reached.

    Thread safety:
        Not thread-safe. FileSense is single-user single-session.
        No locking is required.

    Portability:
        Zero UI framework imports. Safe to instantiate from any context.

    Usage
    -----
        stack = UndoStack()

        # Before accepting a write:
        stack.push(filepath, original_content, instruction="Add docstrings")

        # On undo:
        entry = stack.pop()
        if entry:
            atomic_write(entry["filepath"], entry["content"])

        # UI display:
        count = stack.depth()
        label = stack.peek_label()
    """

    def __init__(self, limit: int | None = None) -> None:
        """
        Parameters
        ----------
        limit : Maximum number of snapshots to retain.
                Defaults to config.UNDO_STACK_LIMIT.
                Oldest entry is silently evicted when limit is reached.
        """
        self._limit: int = limit if limit is not None else config.UNDO_STACK_LIMIT
        self._stack: list[dict] = []

    # ------------------------------------------------------------------
    # WRITE OPERATIONS
    # ------------------------------------------------------------------

    def push(
        self,
        filepath: str,
        content: str,
        instruction: str | None = None,
    ) -> None:
        """
        Push a snapshot onto the undo stack.

        Call this AFTER validation passes and BEFORE atomic_write().
        Never call this for rejected or failed writes.

        Parameters
        ----------
        filepath    : Absolute path of the file being edited.
        content     : Full file content BEFORE the edit is applied.
        instruction : The edit instruction that triggered this change.
                      Optional. Stored for future diff-summary display.
        """
        entry = _make_entry(filepath, content, instruction)
        self._stack.append(entry)

        # Silently evict the oldest entry when cap is reached.
        # List pop(0) is O(n) but stacks are small (≤ UNDO_STACK_LIMIT).
        # No deque needed here.
        while len(self._stack) > self._limit:
            self._stack.pop(0)

    def pop(self) -> dict | None:
        """
        Pop and return the most recent snapshot.

        Returns None if the stack is empty.
        The caller is responsible for calling atomic_write() with the result.

        Returns
        -------
        Snapshot dict:
        {
            "filepath":    str,
            "content":     str,
            "timestamp":   str,
            "instruction": str,
        }
        or None if the stack is empty.
        """
        if not self._stack:
            return None
        return self._stack.pop()

    def clear(self) -> None:
        """
        Discard all snapshots.

        Called when AI memory is wiped or a new session begins.
        """
        self._stack.clear()

    # ------------------------------------------------------------------
    # READ OPERATIONS
    # ------------------------------------------------------------------

    def depth(self) -> int:
        """Return the number of snapshots currently stored."""
        return len(self._stack)

    def is_empty(self) -> bool:
        """Return True if there are no snapshots available."""
        return len(self._stack) == 0

    def peek(self) -> dict | None:
        """
        Return the most recent snapshot without removing it.

        Returns None if the stack is empty.
        Used by the UI to display undo button labels without consuming the entry.
        """
        if not self._stack:
            return None
        return self._stack[-1]

    def peek_label(self) -> str:
        """
        Return a human-readable label for the most recent snapshot.

        Used directly as the undo button label in the UI.

        Returns
        -------
        e.g.  "↩️ Undo: config.py"
              "↩️ Undo: parser.py — Add docstrings"
              "Nothing to undo"
        """
        entry = self.peek()

        if entry is None:
            return "Nothing to undo"

        import os
        filename = os.path.basename(entry["filepath"])
        instruction = entry.get("instruction", "").strip()

        if instruction:
            # Truncate long instructions for display
            display_instr = (
                instruction[:40] + "…"
                if len(instruction) > 40
                else instruction
            )
            return f"↩️ Undo: {filename} — {display_instr}"

        return f"↩️ Undo: {filename}"

    def history(self) -> list[dict]:
        """
        Return a read-only copy of all snapshots, oldest first.

        Used by future history-panel UI features.
        Do not mutate the returned list.
        """
        return list(self._stack)
    