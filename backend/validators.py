"""
Deterministic validation layer for FileSense V1.5.

Core philosophy:
    AI generates text.
    Validators decide whether the output is safe enough to continue.

Responsibilities:
    - reject malformed AI outputs
    - detect catastrophic truncation
    - detect runaway hallucinated expansion
    - perform lightweight language-aware validation
    - return deterministic structured results

Non-responsibilities:
    - NO file writing
    - NO UI rendering
    - NO diff generation
    - NO undo management
    - NO Ollama calls

Run from project root. No sys.path manipulation.
"""

import ast
import json
from pathlib import Path

import config


# ============================================================================
# PUBLIC API
# ============================================================================

def validate_edit(
    original_content: str,
    edited_content: str,
    filepath: str,
) -> dict:
    """
    Validate AI-generated edited content before diff preview or disk write.

    Executes sequentially.
    Rejects early on first failure.

    Validation Order:
        1. Non-empty check
        2. UTF-8 validation
        3. Null-byte validation
        4. Lower-bound ratio check
        5. Upper-bound ratio check
        6. Language-aware validation

    Returns:
        {
            "valid": bool,
            "reason": str | None
        }
    """

    # ----------------------------------------------------------------------
    # Defensive Normalization
    # ----------------------------------------------------------------------

    # Trust-boundary infrastructure must never assume upstream correctness.

    if original_content is None:
        original_content = ""

    if edited_content is None:
        return _reject("Edited content is None.")

    # ----------------------------------------------------------------------
    # 1. Non-empty check
    # ----------------------------------------------------------------------

    if not edited_content.strip():
        return _reject(
            "Edited content is empty or whitespace-only."
        )

    # ----------------------------------------------------------------------
    # 2. UTF-8 validation
    # ----------------------------------------------------------------------

    try:
        edited_content.encode("utf-8")

    except UnicodeEncodeError as e:

        return _reject(
            "UTF-8 validation failed.\n\n"
            f"{type(e).__name__}: {str(e)}"
        )

    # ----------------------------------------------------------------------
    # 3. Null-byte validation
    # ----------------------------------------------------------------------

    # IMPORTANT:
    # Null bytes ARE valid UTF-8.
    # This MUST remain a separate validator.

    if "\x00" in edited_content:
        return _reject(
            "Edited content contains illegal null bytes."
        )

    # ----------------------------------------------------------------------
    # 4. Lower-bound ratio check
    # ----------------------------------------------------------------------

    original_len = len(original_content)
    edited_len = len(edited_content)

    # Skip tiny files — ratio checks become meaningless.

    if original_len >= 50:

        minimum_allowed = max(
            int(original_len * 0.10),
            1,
        )

        if edited_len < minimum_allowed:

            return _reject(
                "Catastrophic truncation detected.\n\n"
                f"Original size : {original_len:,} chars\n"
                f"Edited size   : {edited_len:,} chars\n"
                f"Minimum safe  : {minimum_allowed:,} chars"
            )

    # ----------------------------------------------------------------------
    # 5. Upper-bound ratio check
    # ----------------------------------------------------------------------

    maximum_allowed = max(
        int(original_len * config.MAX_EDIT_SIZE_MULTIPLIER),
        100,
    )

    # Edge-case protection for tiny files.
    # Prevent absurdly small ceilings like:
    #   5-char file × 3.0 = 15-char max

    if edited_len > maximum_allowed:

        return _reject(
            "Runaway generation detected.\n\n"
            f"Original size : {original_len:,} chars\n"
            f"Edited size   : {edited_len:,} chars\n"
            f"Maximum safe  : {maximum_allowed:,} chars"
        )

    # ----------------------------------------------------------------------
    # 6. Language-aware validation
    # ----------------------------------------------------------------------

    extension = Path(filepath).suffix.lower()

    if extension == ".py":

        result = _validate_python(edited_content)

        if not result["valid"]:
            return result

    elif extension == ".json":

        result = _validate_json(edited_content)

        if not result["valid"]:
            return result

    # ----------------------------------------------------------------------
    # SUCCESS
    # ----------------------------------------------------------------------

    return _accept()


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _accept() -> dict:
    """
    Standardized acceptance payload.
    """

    return {
        "valid": True,
        "reason": None,
    }


def _reject(reason: str) -> dict:
    """
    Standardized rejection payload.
    """

    return {
        "valid": False,
        "reason": reason,
    }


# ============================================================================
# LANGUAGE-AWARE VALIDATORS
# ============================================================================

def _validate_python(content: str) -> dict:
    """
    Validate Python syntax using ast.parse().

    Broad exception handling is intentional.

    Trust-boundary code must fail CLOSED.
    """

    try:
        ast.parse(content)

    except Exception as e:

        return _reject(
            "Python syntax validation failed.\n\n"
            f"{type(e).__name__}: {str(e)}"
        )

    return _accept()


def _validate_json(content: str) -> dict:
    """
    Validate JSON structure using json.loads().

    Broad exception handling is intentional for deterministic rejection.
    """

    try:
        json.loads(content)

    except Exception as e:

        return _reject(
            "JSON validation failed.\n\n"
            f"{type(e).__name__}: {str(e)}"
        )

    return _accept()