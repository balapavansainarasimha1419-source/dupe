"""
backend/token_budget.py

Deterministic edit context builder for FileSense V1.

Responsibilities:
  - Estimate token count without any external tokenizer (offline safe).
  - Reject protected paths immediately — before Ollama is ever called.
  - Route: file fits within MAX_FILE_TOKENS → return full-text payload.
  - Route: file too large → return a clean rejection payload.

V1 rule (enforced):
    Only full-file editing is supported. Files exceeding MAX_FILE_TOKENS
    are rejected cleanly with exact word count, token estimate, and limit.
    Protected paths are rejected here — not after waiting 120 seconds for
    Ollama to finish output that will never be written.

V2 note (pre-written, currently bypassed):
    File-type-aware chunking functions are fully implemented at the bottom
    of this file. They are not called anywhere in V1. They will be wired in
    when ChromaDB is redesigned for parent-child embeddings — a substantial
    future migration. DO NOT DELETE.

Design rules:
  - No Streamlit. No PySide6. No XML parsing. No diffs. No file writing.
  - Pure routing and context preparation only.
  - Run from project root. No sys.path manipulation.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import config


# ── File type groups ────────────────────────────────────────────────────────────
#
# SentencePiece (used by Gemma) splits symbols, operators, and camelCase into
# multiple sub-tokens. Code-heavy files therefore produce significantly more
# tokens per word than prose. Using 1.33 on code reliably undercounts and can
# cause silent context overflow on the RTX 2050.

_HIGH_DENSITY_EXTENSIONS = {
    ".py",
    ".js",
    ".json",
    ".ts",
    ".html",
    ".css",
    ".cpp",
    ".java",
    ".go",
    ".rs",
}


# ── Protected path check ────────────────────────────────────────────────────────

def _is_protected(filepath: str) -> bool:
    """
    Return True if the filepath falls inside any of config.PROTECTED_PATHS.

    Uses Path.resolve() for cross-platform absolute path normalization,
    then checks for an exact contiguous component-sequence match.

    This correctly handles multi-segment entries such as "data/chroma_db":
    substring matching would produce false positives on paths like
    "chroma_db_backup/" — component matching does not.

    Fails CLOSED on path-resolution errors. An unresolvable path is treated
    as protected. This aligns with the FileSense safety principle:
        safety > permissiveness

    Does NOT lowercase path parts. Linux filesystems are case-sensitive;
    lowercasing would incorrectly equate "Models/" with "models/".
    """
    try:
        normalized = Path(filepath).resolve()
    except (OSError, RuntimeError):
        # Malformed or inaccessible path — block it.
        # Narrow exception scope is intentional: broad except clauses
        # can silently mask programmer mistakes inside a trust boundary.
        return True

    norm_parts = normalized.parts

    for protected in config.PROTECTED_PATHS:
        # No try/except here — PROTECTED_PATHS is internal trusted config.
        # A bad entry should raise visibly, not be silently skipped.
        # Silent skipping would create undetected protection gaps.
        protected_parts = Path(os.path.normpath(protected)).parts

        if not protected_parts:
            continue

        # Sliding-window contiguous component match.
        p_len = len(protected_parts)
        for i in range(len(norm_parts) - p_len + 1):
            if norm_parts[i : i + p_len] == protected_parts:
                return True

    return False


# ── Token estimation ────────────────────────────────────────────────────────────

def estimate_tokens(text: str, filepath: str = "") -> int:
    """
    Offline token estimate — no tiktoken, no HuggingFace tokenizer required.

    Prose / markdown / plain text : words × 1.33
    Code / config files           : words × 2.00

    The 2.0 multiplier accounts for symbol splitting, indentation tokens,
    and camelCase sub-tokenization in SentencePiece models.

    Known limitation (V1-acceptable):
        Minified or compressed files (e.g. bundled JS, minified JSON) can
        contain very long single "words", causing severe undercounting.
        If unexplained truncation appears on compressed files, the fix is:
            max(len(text.split()), len(text) // 4)
        before applying the multiplier. Not implemented in V1 because
        FileSense users edit readable source files, not minified bundles.
    """
    suffix = Path(filepath).suffix.lower() if filepath else ""
    multiplier = 2.0 if suffix in _HIGH_DENSITY_EXTENSIONS else 1.33
    return int(len(text.split()) * multiplier)


def fits_in_context(text: str, filepath: str = "") -> bool:
    """Return True if the file can be injected in full within MAX_FILE_TOKENS."""
    return estimate_tokens(text, filepath) <= config.MAX_FILE_TOKENS


# ── Section hashing ─────────────────────────────────────────────────────────────

def build_section_hash(text: str) -> str:
    """
    Compute a SHA256 hash of a selected text region.

    Used by the section-level editing system to detect file drift between
    selection time and reinsertion time. If the file changes after the user
    selects a section, the hash comparison fails and reinsertion is rejected
    safely rather than overwriting incorrect line ranges.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Main entry point ────────────────────────────────────────────────────────────

def build_edit_context(
    filepath: str,
    text: str,
) -> dict:
    """
    Build the AI-ready context payload for ollama_bridge.generate_file_edit().

    Parameters
    ----------
    filepath : Absolute path to the file being edited.
    text     : Complete file content read from disk.

    Returns
    -------
    A unified payload dict consumed by the UI and the bridge:

    {
        "mode":              "full" | "rejected",
        "editable":          bool,
        "filepath":          str,
        "content":           str | None,
        "word_count":        int,
        "estimated_tokens":  int,
        "reason":            str | None,
    }

    Caller contract
    ---------------
    "rejected" → display payload["reason"] to the user; do NOT call Ollama.
    "full"     → pass payload to ollama_bridge.generate_file_edit().

    Both "mode" and "editable" are intentionally present:
        - "editable" (bool) is for simple UI conditionals right now.
        - "mode" (str)  is for future extensibility — "section", "readonly",
          and other routing states will use this field in V2.
    """
    word_count = len(text.split())
    estimated_tokens = estimate_tokens(text, filepath)
    suffix = Path(filepath).suffix.lower() if filepath else ""

    # ── Guard 1: protected path ────────────────────────────────────────────────
    # Rejected here — not only in file_writer — so the user is told immediately,
    # before Ollama spends 60–120 seconds on output that will be discarded.
    if _is_protected(filepath):
        return {
            "mode": "rejected",
            "editable": False,
            "filepath": filepath,
            "content": None,
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
            "reason": (
                f"'{os.path.basename(filepath)}' is inside a protected "
                f"directory and cannot be edited.\n\n"
                f"Protected paths:\n  - "
                + "\n  - ".join(config.PROTECTED_PATHS)
            ),
        }

    # ── Guard 2: file too large ────────────────────────────────────────────────
    if not fits_in_context(text, filepath):
        density_note = (
            " (code files use a 2× token multiplier for syntax overhead)"
            if suffix in _HIGH_DENSITY_EXTENSIONS
            else ""
        )
        return {
            "mode": "rejected",
            "editable": False,
            "filepath": filepath,
            "content": None,
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
            "reason": (
                f"This file is too large for safe full-file editing"
                f"{density_note}.\n"
                f"  Words            : {word_count:,}\n"
                f"  Estimated tokens : {estimated_tokens:,}\n"
                f"  Safe limit       : {config.MAX_FILE_TOKENS:,} tokens\n\n"
                f"Try selecting a smaller section to edit instead."
            ),
        }

    # ── Full injection path ────────────────────────────────────────────────────
    # "mode" is "full" only in V1. Section editing mode is intentionally
    # absent here — reinsertion logic lives in a separate subsystem and
    # will extend this routing in V2. Do not add section routing here yet.
    return {
        "mode": "full",
        "editable": True,
        "filepath": filepath,
        "content": text,
        "word_count": word_count,
        "estimated_tokens": estimated_tokens,
        "reason": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# V2 CHUNKING LOGIC — PRE-WRITTEN, CURRENTLY BYPASSED
# ══════════════════════════════════════════════════════════════════════════════
#
# These functions are complete and correct. They are not called anywhere in V1.
# They will be wired into build_edit_context() in V2 once ChromaDB is
# redesigned for parent-child chunk embeddings per file.
#
# DO NOT DELETE. DO NOT CALL FROM V1 CODE.
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    filepath: str = "",
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """
    Split text using a file-type-aware strategy.

    .py         → function/class boundary splitting
    .md / .rst  → heading boundary splitting
    .txt        → paragraph splitting
    fallback    → fixed-size with word overlap
    """
    ext = Path(filepath).suffix.lower() if filepath else ""

    if ext == ".py":
        chunks = _chunk_python(text)
    elif ext in (".md", ".rst"):
        chunks = _chunk_markdown(text)
    elif ext == ".txt":
        chunks = _chunk_paragraphs(text)
    else:
        chunks = _chunk_fixed(text, chunk_size, overlap)

    return [c.strip() for c in chunks if c.strip()]


def _chunk_python(text: str) -> list[str]:
    """
    Split at top-level def/class boundaries.

    The lookahead uses (?:\\n|^) instead of just \\n to correctly handle
    files where the first function or class begins on line 1 with no
    leading whitespace or module docstring.
    """
    parts = re.split(r"(?=(?:\n|^)(?:def |class ))", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks if chunks else [text]


def _chunk_markdown(text: str) -> list[str]:
    """Split at heading boundaries (# ## ### etc.)."""
    parts = re.split(r"(?=\n#{1,6} )", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks if chunks else [text]


def _chunk_paragraphs(text: str) -> list[str]:
    """Split on blank lines."""
    parts = re.split(r"\n\s*\n", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks if chunks else [text]


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fixed-size chunking with word-level overlap. Universal fallback."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks
