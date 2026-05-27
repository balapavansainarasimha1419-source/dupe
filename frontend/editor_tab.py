"""
frontend/editor_tab.py

Streamlit render function for Tab 3 — AI Editor.

Responsibilities:
    - render Edit and Explain panels (idle stage)
    - render diff review and accept/reject (diff_review stage)
    - render undo controls
    - orchestrate the edit pipeline in correct order

Pipeline order enforced here:
    build_edit_context()        [token_budget.py]
        ↓
    generate_file_edit()        [ollama_bridge.py]
        ↓
    validate_edit()             [validators.py]
        ↓
    build_diff_payload()        [diff_preview.py]
        ↓
    [user decision]
        ↓
    undo_stack.push()           [undo_stack.py]
        ↓
    atomic_write()              [file_writer.py]

Boundary rules:
    - renders diffs.     Does NOT compute them.
    - triggers undo.     Does NOT store snapshots.
    - calls write.       Does NOT manage atomicity.

Non-responsibilities:
    - NO diff math
    - NO snapshot storage
    - NO filesystem mutation outside atomic_write()
    - NO Ollama calls directly
"""

from __future__ import annotations

import os

import streamlit as st

import config
from backend.ollama_bridge import generate_file_edit, explain_code_snippet
from backend.token_budget import build_edit_context, estimate_tokens
from backend.validators import validate_edit
from backend.file_writer import atomic_write
from backend.diff_preview import build_diff_payload


# ============================================================================
# STATE INITIALISATION
#
# Called once at app startup via init_state().
# Keeps session state keys scoped here — app.py doesn't need to know them.
# ============================================================================

def init_state() -> None:
    """
    Initialise all Tab 3 session state keys.

    Must be called from app.py before st.tabs() is rendered.
    Safe to call on every rerun — only sets keys that don't exist yet.
    """
    defaults = {
        "editor_stage":            "idle",
        "editor_edit_context":     None,
        "editor_bridge_result":    None,
        "editor_original_content": None,
        "editor_diff_payload":     None,
        "editor_instruction":      "",
        "editor_explain_output":   None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ============================================================================
# PRIVATE: STATE RESET
# ============================================================================

def _reset() -> None:
    """
    Reset all editor state keys to idle.

    Called before every st.rerun() that returns to the idle stage.
    Centralised here so future key additions only need one update.
    """
    st.session_state.editor_stage            = "idle"
    st.session_state.editor_edit_context     = None
    st.session_state.editor_bridge_result    = None
    st.session_state.editor_original_content = None
    st.session_state.editor_diff_payload     = None
    st.session_state.editor_instruction      = ""


# ============================================================================
# PUBLIC: RENDER ENTRY POINT
# ============================================================================

def render(BACKEND_READY: bool) -> None:
    """Render the AI Editor tab. Called from app.py inside `with tab_editor:`."""

    engine_name = config.OLLAMA_MODEL if BACKEND_READY else "AI"
    st.header("🤖 Offline AI Editor")
    st.caption(f"Powered by local **{engine_name}** · All edits stay on your machine.")

    if not BACKEND_READY:
        st.error("Backend not available. Cannot use the AI Editor.")
        return

    _render_undo_controls()
    st.divider()

    if st.session_state.editor_stage == "idle":
        _render_idle_stage(engine_name)
    elif st.session_state.editor_stage == "diff_review":
        _render_diff_review_stage()


# ============================================================================
# PRIVATE: UNDO CONTROLS
# ============================================================================

def _render_undo_controls() -> None:
    undo_stack = st.session_state.undo_stack

    if undo_stack.is_empty():
        return

    undo_col, info_col = st.columns([2, 3])

    with undo_col:
        if st.button(undo_stack.peek_label(), use_container_width=True, key="undo_btn"):
            entry = undo_stack.pop()
            if entry:
                result = atomic_write(entry["filepath"], entry["content"])
                if result["success"]:
                    st.toast(f"↩️ Undone: {os.path.basename(entry['filepath'])}")
                else:
                    # Write failed — push the entry back so history isn't lost
                    undo_stack.push(
                        entry["filepath"],
                        entry["content"],
                        entry.get("instruction"),
                    )
                    st.error(f"Undo write failed: {result['error']}")
            _reset()
            st.rerun()

    with info_col:
        st.caption(
            f"**{undo_stack.depth()}** snapshot(s) · limit {config.UNDO_STACK_LIMIT}"
        )


# ============================================================================
# PRIVATE: IDLE STAGE
# ============================================================================

def _render_idle_stage(engine_name: str) -> None:
    edit_col, explain_col = st.columns([3, 2])

    with edit_col:
        _render_edit_panel(engine_name)

    with explain_col:
        _render_explain_panel()


def _render_edit_panel(engine_name: str) -> None:
    st.subheader("✏️ Edit a File")

    editor_filepath = st.text_input(
        "File path",
        placeholder="/absolute/path/to/your/file.py",
        key="editor_filepath_input",
        help="Paste the absolute path to the file you want to edit.",
    )

    editor_instruction = st.text_area(
        "Edit instruction",
        placeholder="e.g. Add docstrings to all functions.",
        height=120,
        key="editor_instruction_input",
    )

    # Live token budget preview — estimate_tokens() does the math
    if editor_filepath and os.path.isfile(editor_filepath):
        try:
            with open(editor_filepath, "r", encoding="utf-8", errors="ignore") as _f:
                _preview_text = _f.read()
            _est   = estimate_tokens(_preview_text, editor_filepath)
            _words = len(_preview_text.split())
            _pct   = min(100, int((_est / config.MAX_FILE_TOKENS) * 100))
            st.caption(
                f"📊 **{_words:,} words** · "
                f"~**{_est:,} tokens** · "
                f"limit {config.MAX_FILE_TOKENS:,} · "
                f"**{_pct}% used**"
            )
            if _pct >= 100:
                st.warning("⚠️ File exceeds the safe token limit and will be rejected.")
        except Exception:
            pass
    elif editor_filepath and not os.path.isfile(editor_filepath):
        st.caption("⚠️ Path does not point to an existing file.")

    if not st.button(
        "🚀 Generate Edit",
        type="primary",
        use_container_width=True,
        disabled=not (
            editor_filepath
            and editor_filepath.strip()
            and editor_instruction
            and editor_instruction.strip()
        ),
    ):
        return

    fp    = editor_filepath.strip()
    instr = editor_instruction.strip()

    if not os.path.isfile(fp):
        st.error(f"File not found: {fp}")
        return

    try:
        with open(fp, "r", encoding="utf-8", errors="ignore") as _f:
            file_text = _f.read()
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    # Step 1 — Token budget
    with st.spinner("Checking file size…"):
        edit_context = build_edit_context(fp, file_text)

    if not edit_context["editable"]:
        st.error(edit_context["reason"])
        return

    # Step 2 — Generate
    with st.spinner(
        f"🧠 {engine_name} is editing `{os.path.basename(fp)}`… "
        f"(up to {config.OLLAMA_EDIT_TIMEOUT}s)"
    ):
        bridge_result = generate_file_edit(edit_context, instr)

    if not bridge_result["success"]:
        st.error(f"AI edit failed: {bridge_result['error']}")
        return

    # Step 3 — Validate
    val_result = validate_edit(
        original_content=file_text,
        edited_content=bridge_result["extracted_content"],
        filepath=fp,
    )

    if not val_result["valid"]:
        st.error("Validation rejected the AI output:\n\n" + val_result["reason"])
        return

    # Step 4 — Build diff payload (backend computes, UI will only render)
    diff_payload = build_diff_payload(
        original=file_text,
        edited=bridge_result["extracted_content"],
        filepath=fp,
    )

    # Persist all state and transition
    st.session_state.editor_edit_context     = edit_context
    st.session_state.editor_bridge_result    = bridge_result
    st.session_state.editor_original_content = file_text
    st.session_state.editor_diff_payload     = diff_payload
    st.session_state.editor_instruction      = instr
    st.session_state.editor_stage            = "diff_review"
    st.rerun()


def _render_explain_panel() -> None:
    """Read-only explain panel. Zero write risk. Outside all validation/undo/diff."""
    st.subheader("🔍 Explain a Snippet")
    st.caption(
        "Paste any code or text for a plain-language explanation. "
        "Nothing is written to disk."
    )

    explain_filepath = st.text_input(
        "Source file (optional)",
        placeholder="/path/to/file.py  ← gives AI context",
        key="explain_filepath_input",
    )

    explain_snippet = st.text_area(
        "Paste snippet here",
        height=160,
        placeholder="Paste the code or text you want explained…",
        key="explain_snippet_input",
    )

    if st.button(
        "💡 Explain",
        use_container_width=True,
        disabled=not (explain_snippet and explain_snippet.strip()),
        key="explain_btn",
    ):
        with st.spinner("🧠 Explaining…"):
            explanation = explain_code_snippet(
                snippet=explain_snippet.strip(),
                filepath=explain_filepath.strip(),
            )
        st.session_state.editor_explain_output = explanation

    if st.session_state.editor_explain_output:
        st.divider()
        st.markdown("**Explanation:**")
        st.write(st.session_state.editor_explain_output)
        if st.button("✕ Clear", key="clear_explain"):
            st.session_state.editor_explain_output = None
            st.rerun()


# ============================================================================
# PRIVATE: DIFF REVIEW STAGE
# ============================================================================

def _render_diff_review_stage() -> None:
    """Render pre-computed diff payload. No math. UI reads, never recomputes."""

    diff     = st.session_state.editor_diff_payload
    fp       = diff["filepath"]
    original = diff["original"]
    edited   = diff["edited"]

    st.subheader(f"Review Edit — `{os.path.basename(fp)}`")
    st.caption(fp)

    # All stats come from the payload
    delta_sign = "+" if diff["chars_delta"] >= 0 else ""
    st.caption(
        f"Lines: **{diff['lines_original']}** → **{diff['lines_edited']}** · "
        f"**+{diff['lines_added']}** added · "
        f"**−{diff['lines_removed']}** removed · "
        f"**{diff['lines_unchanged']}** unchanged · "
        f"chars: {delta_sign}{diff['chars_delta']:,}"
    )

    if diff["is_identical"]:
        st.info(
            "The AI returned content identical to the original. "
            "No changes to accept."
        )

    # Diff viewer — renders the payload
    try:
        from streamlit_diff_viewer import diff_viewer
        diff_viewer(original, edited, split_view=True)
    except ImportError:
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**Original**")
            st.code(original, language="text")
        with dc2:
            st.markdown("**Edited**")
            st.code(edited, language="text")

    st.divider()

    accept_col, reject_col, _ = st.columns([2, 2, 4])

    with accept_col:
        if st.button(
            "✅ Accept & Write",
            type="primary",
            use_container_width=True,
            disabled=diff["is_identical"],
        ):
            instr      = st.session_state.get("editor_instruction", "")
            undo_stack = st.session_state.undo_stack

            # Push snapshot BEFORE writing.
            # Failed writes must NOT enter undo history — handled below.
            undo_stack.push(fp, original, instruction=instr)

            write_result = atomic_write(fp, edited)

            if write_result["success"]:
                st.toast(f"✅ Saved: {os.path.basename(fp)}")
                _reset()
                st.rerun()
            else:
                # Write failed — pop the snapshot we just pushed
                undo_stack.pop()
                st.error(f"Write failed: {write_result['error']}")

    with reject_col:
        if st.button("❌ Reject", use_container_width=True):
            _reset()
            st.rerun()
            