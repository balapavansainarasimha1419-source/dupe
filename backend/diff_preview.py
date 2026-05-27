"""
Deterministic diff payload generator for FileSense V1.5.

Core philosophy:
    The UI renders.
    The backend calculates.

Responsibilities:
    - compute unified diff between original and edited content
    - compute line-level statistics (added, removed, unchanged)
    - return a frozen, serializable payload the UI consumes directly
    - never touch disk
    - never call Ollama
    - never render anything

Non-responsibilities (enforced, not aspirational):
    - NO file writing
    - NO Streamlit / PySide6
    - NO validators
    - NO Ollama calls
    - NO undo management

Design rule:
    The UI should be able to swap from Streamlit to PySide6 and consume
    the exact same diff payload without any changes to this module.
    That portability is the point of isolating diff logic here.

Run from project root. No sys.path manipulation.
"""

from __future__ import annotations

import difflib


# ============================================================================
# PUBLIC API
# ============================================================================

def build_diff_payload(
    original: str,
    edited: str,
    filepath: str = "",
) -> dict:
    """
    Compute a complete diff payload between original and edited content.

    Parameters
    ----------
    original : The original file content before AI editing.
    edited   : The AI-generated edited content, post-validation.
    filepath : Optional. Used to label the diff header lines.

    Returns
    -------
    FROZEN CONTRACT — UI components depend on these exact keys:

    {
        "original":        str,        # original content, unchanged
        "edited":          str,        # edited content, unchanged
        "unified_diff":    str,        # full unified diff string
        "lines_original":  int,        # line count of original
        "lines_edited":    int,        # line count of edited
        "lines_added":     int,        # lines present in edited, not in original
        "lines_removed":   int,        # lines present in original, not in edited
        "lines_unchanged": int,        # lines identical in both
        "chars_original":  int,        # character count of original
        "chars_edited":    int,        # character count of edited
        "chars_delta":     int,        # chars_edited - chars_original (signed)
        "is_identical":    bool,       # True if no changes were detected
        "filepath":        str,        # filepath label passed in
    }

    The UI must consume this payload directly.
    The UI must NOT recompute any of these values itself.

    Caller contract
    ---------------
    Call ONLY after validators.validate_edit() has returned valid=True.
    This function does not re-validate. It trusts the caller.
    """

    # ------------------------------------------------------------------
    # Defensive normalization
    # Trust-boundary infrastructure must never assume upstream correctness.
    # ------------------------------------------------------------------

    if original is None:
        original = ""
    if edited is None:
        edited = ""

    original_lines = original.splitlines(keepends=True)
    edited_lines   = edited.splitlines(keepends=True)

    # ------------------------------------------------------------------
    # Unified diff
    #
    # fromfile / tofile labels use filepath for readable output.
    # lineterm="" suppresses extra newlines — difflib adds them by default,
    # which causes double-spacing when rendered in most UI components.
    # ------------------------------------------------------------------

    label = filepath or "file"

    unified = difflib.unified_diff(
        original_lines,
        edited_lines,
        fromfile=f"{label} (original)",
        tofile=f"{label} (edited)",
        lineterm="",
    )

    unified_diff_str = "\n".join(unified)

    # ------------------------------------------------------------------
    # Line-level statistics
    #
    # Uses SequenceMatcher for accurate added/removed/unchanged counts.
    # The naive approach (set subtraction) double-counts duplicate lines —
    # e.g. if "pass" appears 5 times in original and 4 in edited, set
    # subtraction reports 0 removed. SequenceMatcher handles this correctly.
    # ------------------------------------------------------------------

    matcher = difflib.SequenceMatcher(
        None,
        original.splitlines(),
        edited.splitlines(),
        autojunk=False,  # autojunk can suppress small real changes
    )

    lines_added     = 0
    lines_removed   = 0
    lines_unchanged = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            lines_unchanged += i2 - i1
        elif tag == "insert":
            lines_added += j2 - j1
        elif tag == "delete":
            lines_removed += i2 - i1
        elif tag == "replace":
            lines_removed += i2 - i1
            lines_added   += j2 - j1

    chars_original = len(original)
    chars_edited   = len(edited)

    return {
        "original":        original,
        "edited":          edited,
        "unified_diff":    unified_diff_str,
        "lines_original":  len(original.splitlines()),
        "lines_edited":    len(edited.splitlines()),
        "lines_added":     lines_added,
        "lines_removed":   lines_removed,
        "lines_unchanged": lines_unchanged,
        "chars_original":  chars_original,
        "chars_edited":    chars_edited,
        "chars_delta":     chars_edited - chars_original,
        "is_identical":    original == edited,
        "filepath":        filepath,
    }
