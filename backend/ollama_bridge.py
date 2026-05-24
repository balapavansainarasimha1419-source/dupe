"""
backend/ollama_bridge.py

Deterministic Ollama bridge for FileSense V1.

This module is a TRUST-BOUNDARY layer between:
    - untrusted LLM output
    - the validated editing pipeline

Core philosophy:
    filesystem = source of truth
    AI = untrusted text generator

Responsibilities:
    - communicate with Ollama safely
    - enforce deterministic edit prompting
    - enforce XML output contract
    - extract recoverable XML output
    - isolate explain/chat mode from edit mode

Non-responsibilities (enforced, not aspirational):
    - NO file writing            (file_writer.py owns all disk writes)
    - NO diffs                   (difflib in Tab 3 owns all diffs)
    - NO validators embedded     (validators.py owns all correctness checks)
    - NO undo stack management   (app.py Tab 3 owns undo)
    - NO Streamlit / PySide6

Run from project root. No sys.path manipulation.

---

FINALIZED WORKLOAD TEMPERATURE STRATIFICATION:

    generate_file_edit()   → config.OLLAMA_TEMPERATURE        (0.1)
        XML discipline, stable regeneration, low hallucination rate.

    ask_local_ai()         → config.OLLAMA_CHAT_TEMPERATURE    (0.2)
        Factual conversational Q&A. Slightly above edit floor for fluency.

    explain_code_snippet() → config.OLLAMA_EXPLAIN_TEMPERATURE (0.3)
        Plain language explanation. Fluency matters more than rigid discipline.

These are intentionally different values. Do not collapse them to one constant.
"""

from __future__ import annotations

import logging
import re

import requests

import config


# ============================================================================
# OLLAMA ENDPOINT
# ============================================================================
#
# FINALIZED:
#   MUST use /api/generate    (not /api/chat)
#   MUST read data.get("response")    (not data["message"]["content"])
#
OLLAMA_API_URL = "http://localhost:11434/api/generate"


# ============================================================================
# PRIVATE: SMART CONTEXT TRUNCATION
# ============================================================================

def _smart_truncate(text: str, max_words: int) -> str:
    """
    Safely truncate conversational context near sentence boundaries.

    SCOPE — explain/chat mode ONLY.

    This helper must NEVER be used for deterministic file editing.
    Editing requires full-file integrity. Files that are too large for
    the editing context window are rejected early by token_budget.py,
    not silently truncated here.
    """
    words = text.split()

    if len(words) <= max_words:
        return text

    chopped = " ".join(words[:max_words])

    last_period = chopped.rfind(".")
    last_newline = chopped.rfind("\n")
    break_point = max(last_period, last_newline)

    if break_point > 0:
        return (
            chopped[:break_point + 1]
            + "\n\n... [Context Truncated for Memory Limits]"
        )

    return chopped + "..."


# ============================================================================
# PRIVATE: XML EXTRACTION PIPELINE
# ============================================================================

def _extract_filesense_xml(raw_response: str) -> str | None:
    """
    Extract edited file content from the XML output contract.

    FINALIZED CONTRACT:

        <filesense_output>
        FULL FILE CONTENT HERE
        </filesense_output>

    Extraction runs three passes, in order.

    Pass 1 — Direct extraction
        Regex on the raw response. No cleaning.
        Handles well-behaved model output.

    Pass 2 — Strip markdown fences, retry
        Handles the most common deviation: Gemma wrapping the XML
        inside a ```python or ``` code block.
        Strips all fence variants (with or without language hint) then
        repeats Pass 1.

    Pass 3 — Missing closing-tag recovery
        Handles: model opened <filesense_output> but truncated before
        the closing tag. Recovers everything after the opening tag.

        SAFETY GATE:
            recovered length > 50% of cleaned response length.

        Rationale: if the recovered portion is less than half the total
        response, the model wrote mostly preamble and very little content.
        That pattern indicates hallucination or severe truncation — reject.
        Validators remain the final authority regardless of what this
        function returns.

    Returns:
        Extracted content string (stripped), or None if all passes fail.

    IMPORTANT:
        This function extracts only.
        It does NOT validate syntax, AST integrity, encoding,
        size ratios, or semantic correctness.
        All of that is validators.py's job.
    """
    if not raw_response or not raw_response.strip():
        return None

    # \s* inside the pattern handles leading/trailing newlines and
    # model indentation after the opening tag. .strip() on group(1)
    # catches anything \s* misses. Both layers are intentional.
    pattern = re.compile(
        r"<filesense_output>\s*(.*?)\s*</filesense_output>",
        re.DOTALL,
    )

    # ------------------------------------------------------------------
    # Pass 1 — Direct extraction
    # ------------------------------------------------------------------

    match = pattern.search(raw_response)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted

    # ------------------------------------------------------------------
    # Pass 2 — Strip markdown fences, retry
    # ------------------------------------------------------------------
    #
    # Pattern covers:
    #   ```python, ```xml, ```json, ```typescript, ``` (bare)
    # \n? handles optional trailing newline after fence.
    #
    cleaned = re.sub(r"```[a-zA-Z]*\n?", "", raw_response).strip()
    cleaned = cleaned.replace("```", "").strip()

    match = pattern.search(cleaned)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted

    # ------------------------------------------------------------------
    # Pass 3 — Guarded missing closing-tag recovery
    # ------------------------------------------------------------------

    start_tag = "<filesense_output>"
    start_index = cleaned.find(start_tag)

    if start_index != -1:
        recovered = cleaned[start_index + len(start_tag):].strip()

        if recovered and len(cleaned) > 0:
            # FINALIZED SAFETY GATE — see docstring for rationale
            if len(recovered) > len(cleaned) * 0.5:
                return recovered

    return None


# ============================================================================
# CONVERSATIONAL / CHAT MODE
# ============================================================================

def ask_local_ai(prompt: str, context_text: str = "") -> str:
    """
    Conversational Ollama bridge for file Q&A and search assistance.

    ISOLATION CONTRACT:
        Does NOT use XML extraction.
        Does NOT validate outputs.
        Does NOT write files.
        Does NOT interact with the undo stack.
        Does NOT enforce deterministic regeneration.

    Temperature: config.OLLAMA_CHAT_TEMPERATURE (0.2)
        Slightly above edit floor. Factual Q&A benefits from
        fluency without needing rigid XML discipline.

    Timeout: config.INFERENCE_TIMEOUT
        Chat-scale workload, not full-file regeneration.

    Returns:
        Human-readable AI response string.
        Never raises — errors are returned as user-facing strings.
    """
    if context_text.strip():
        context_text = _smart_truncate(context_text, config.MAX_CONTEXT_WORDS)

        system_prompt = (
            "You are the FileSense AI.\n\n"
            "Use the provided file snippets to answer the user's question "
            "accurately.\n\n"
            "If the answer is not present in the snippets, say so clearly "
            "instead of inventing information.\n\n"
            "--- FILE CONTEXT ---\n"
            f"{context_text}\n\n"
            "--- USER QUESTION ---\n"
            f"{prompt}"
        )
    else:
        system_prompt = prompt

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": False,
        "options": {
            "temperature": config.OLLAMA_CHAT_TEMPERATURE,
            "top_p": config.OLLAMA_TOP_P,
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=config.INFERENCE_TIMEOUT,
        )

        if response.status_code == 404:
            return (
                "🚨 AI Model Not Found!\n\n"
                "Ollama is running but the configured model has not been "
                "downloaded yet.\n\n"
                f"Run:\n    ollama run {config.OLLAMA_MODEL}"
            )

        response.raise_for_status()

        data = response.json()
        return data.get("response", "Error: Empty response from AI.")

    except requests.exceptions.ConnectionError:
        return (
            "🚨 Connection Failed: Could not connect to Ollama.\n\n"
            "Check:\n"
            "1. Ollama is installed (ollama.com)\n"
            "2. Ollama is running in your system tray\n"
            f"3. The model exists (ollama run {config.OLLAMA_MODEL})"
        )
    except requests.exceptions.Timeout:
        return "⚠️ Timeout: The local AI took too long to respond."
    except Exception as e:
        logging.error(f"ask_local_ai() unexpected error: {e}")
        return f"⚠️ Unexpected Error: {str(e)}"


# ============================================================================
# EXPLAIN MODE
# ============================================================================

def explain_code_snippet(snippet: str, filepath: str = "") -> str:
    """
    Ask Ollama to explain a selected section of a file in plain language.

    ISOLATION CONTRACT (enforced, not aspirational):
        Does NOT use XML extraction.
        Does NOT run validators.
        Does NOT write files.
        Does NOT touch the undo stack.
        Zero write risk by design.

    Temperature: config.OLLAMA_EXPLAIN_TEMPERATURE (0.3)
        Higher than edit and chat modes. Explanation generation benefits
        from fluency. Rigid low temperature produces robotic output here.
        This is intentional — do not "correct" it to OLLAMA_TEMPERATURE.

    Timeout: config.INFERENCE_TIMEOUT
        Explanation is a chat-scale workload, not full-file regeneration.
        Separate from OLLAMA_EDIT_TIMEOUT intentionally.

    Parameters
    ----------
    snippet  : The text the user selected for explanation.
    filepath : Optional. Gives the model file-type context
               (e.g. "from config.py" tells it to expect Python).

    Returns
    -------
    Plain text explanation string.
    Returns a user-facing error message string on failure.
    Never raises.
    """
    file_hint = f" from `{filepath}`" if filepath else ""

    prompt = (
        f"Explain the following code or text{file_hint} clearly and concisely.\n\n"
        f"Cover:\n"
        f"- What it does\n"
        f"- Why it exists (its purpose in context)\n"
        f"- Any important implementation details a developer should know\n\n"
        f"Write for an intermediate developer. No preamble.\n\n"
        f"---\n{snippet}\n---"
    )

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.OLLAMA_EXPLAIN_TEMPERATURE,
            "top_p": config.OLLAMA_TOP_P,
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=config.INFERENCE_TIMEOUT,
        )

        if response.status_code == 404:
            return (
                f"⚠️ Model '{config.OLLAMA_MODEL}' not found. "
                f"Run: ollama run {config.OLLAMA_MODEL}"
            )

        response.raise_for_status()

        data = response.json()
        return data.get("response", "Error: Empty response from AI.")

    except requests.exceptions.ConnectionError:
        return "⚠️ Could not connect to Ollama. Is the app running in your system tray?"
    except requests.exceptions.Timeout:
        return "⚠️ Explanation request timed out. Try selecting a smaller snippet."
    except Exception as e:
        logging.error(f"explain_code_snippet() unexpected error: {e}")
        return f"⚠️ Unexpected Error: {str(e)}"


# ============================================================================
# DETERMINISTIC FILE EDITING
# ============================================================================

def generate_file_edit(edit_context: dict, instruction: str) -> dict:
    """
    Generate a deterministic full-file edit candidate.

    Consumes the frozen payload contract from:
        token_budget.build_edit_context()

    ISOLATION CONTRACT:
        Does NOT validate syntax, AST integrity, JSON structure, or encoding.
        Does NOT write files.
        Does NOT compute diffs.
        Does NOT manage the undo stack.
        All of those responsibilities belong to downstream components.

    Parameters
    ----------
    edit_context : Frozen payload from token_budget.build_edit_context().
                   Caller MUST verify payload["editable"] is True before calling.
    instruction  : The user's edit instruction.

    Returns
    -------
    FROZEN CONTRACT — downstream components depend on these exact keys:

    Success:
    {
        "success":           True,
        "extracted_content": str,
        "raw_response":      str,
        "error":             None,
    }

    Failure:
    {
        "success":           False,
        "extracted_content": None,
        "raw_response":      str | None,
        "error":             str,
    }

    Caller contract
    ---------------
    "success": False → show result["error"] to user. Do NOT proceed to write.
    "success": True  → pass result["extracted_content"] to validators.py,
                       then show diff, then call file_writer.py on user accept.

    Undo stack
    ----------
    The CALLER (Tab 3 UI) pushes the current file content onto the undo stack
    BEFORE accepting the write. This function never touches the undo stack.
    """

    # -------------------------------------------------------------------------
    # Guard 1 — editable flag
    #
    # Checked here AND in the UI — two independent layers, both intentional.
    # Calling this with a non-editable payload is a programmer error.
    # -------------------------------------------------------------------------

    if not edit_context.get("editable"):
        return {
            "success": False,
            "extracted_content": None,
            "raw_response": None,
            "error": (
                edit_context.get("reason")
                or "Editing rejected by token budget system."
            ),
        }

    # -------------------------------------------------------------------------
    # Guard 2 — mode check
    #
    # V1 supports "full" only. When V2 introduces "section" and "readonly",
    # those modes will have their own routing. This guard ensures unrecognized
    # future modes fail loudly rather than silently falling through.
    # -------------------------------------------------------------------------

    if edit_context.get("mode") != "full":
        return {
            "success": False,
            "extracted_content": None,
            "raw_response": None,
            "error": (
                f"Unsupported edit mode: '{edit_context.get('mode')}'. "
                f"Only 'full' is supported in V1."
            ),
        }

    # -------------------------------------------------------------------------
    # Guard 3 — defensive content validation
    #
    # Direct key access would raise KeyError on a malformed payload.
    # Trust-boundary code fails gracefully, not loudly.
    # -------------------------------------------------------------------------

    original_content = edit_context.get("content")

    if not original_content or not original_content.strip():
        return {
            "success": False,
            "extracted_content": None,
            "raw_response": None,
            "error": "Edit context contained no file content.",
        }

    filepath = edit_context.get("filepath", "unknown")
    word_count = edit_context.get("word_count", 0)
    token_estimate = edit_context.get("estimated_tokens", 0)

    # -------------------------------------------------------------------------
    # Deterministic edit prompt
    #
    # Rule ordering matters: Gemma 4B follows earlier rules more reliably
    # under long context pressure. "Return the COMPLETE file" is Rule 1.
    #
    # The XML contract is stated once as a rule and once as a concrete example.
    # Repetition reduces closing-tag omission — the most common failure mode
    # on constrained hardware.
    #
    # File content is delimiter-wrapped to reduce context blur between
    # instructions and file body under long inputs.
    # -------------------------------------------------------------------------

    system_prompt = (
        f"You are the FileSense deterministic file editor.\n\n"
        f"FILE: {filepath}\n"
        f"SIZE: {word_count} words (~{token_estimate} estimated tokens)\n\n"
        f"EDIT INSTRUCTION:\n{instruction}\n\n"
        f"RULES — follow all of them exactly:\n"
        f"1. Return the COMPLETE edited file. Do not truncate, summarize, or omit any section.\n"
        f"2. Apply ONLY the changes the instruction asks for. Leave everything else identical.\n"
        f"3. Do NOT add explanations, commentary, or markdown fences.\n"
        f"4. Wrap your ENTIRE output in these XML tags and nothing else:\n\n"
        f"   <filesense_output>\n"
        f"   ...complete file content here...\n"
        f"   </filesense_output>\n\n"
        f"5. Close the </filesense_output> tag. Do not stop before closing it.\n\n"
        f"--- BEGIN FILE ---\n"
        f"{original_content}\n"
        f"--- END FILE ---"
    )

    payload = {
        "model": config.OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": False,
        "options": {
            "temperature": config.OLLAMA_TEMPERATURE,  # 0.1 — XML discipline
            "top_p": config.OLLAMA_TOP_P,              # 0.9
            "num_ctx": config.OLLAMA_NUM_CTX,          # 8192 — tuned for RTX 2050 VRAM
        },
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=config.OLLAMA_EDIT_TIMEOUT,  # Separate from INFERENCE_TIMEOUT.
                                                  # Editing = full-file regeneration.
                                                  # Chat = short-form response.
                                                  # These are different workloads.
        )

        # 404 = model not downloaded yet
        if response.status_code == 404:
            return {
                "success": False,
                "extracted_content": None,
                "raw_response": None,
                "error": (
                    f"Model '{config.OLLAMA_MODEL}' not found in Ollama. "
                    f"Run: ollama pull {config.OLLAMA_MODEL}"
                ),
            }

        response.raise_for_status()

        data = response.json()
        raw_response = data.get("response", "")

        if not raw_response.strip():
            return {
                "success": False,
                "extracted_content": None,
                "raw_response": raw_response,
                "error": "Ollama returned an empty response. The model may have stalled.",
            }

        # Run the 3-pass XML extraction pipeline
        extracted = _extract_filesense_xml(raw_response)

        if extracted is None:
            return {
                "success": False,
                "extracted_content": None,
                "raw_response": raw_response,
                "error": (
                    "AI output did not contain a valid <filesense_output> block. "
                    "All three extraction passes failed."
                ),
            }

        return {
            "success": True,
            "extracted_content": extracted,
            "raw_response": raw_response,
            "error": None,
        }

    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "extracted_content": None,
            "raw_response": None,
            "error": (
                "Could not connect to Ollama. "
                "Is the Ollama app running in your system tray?"
            ),
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "extracted_content": None,
            "raw_response": None,
            "error": (
                f"Edit request timed out after {config.OLLAMA_EDIT_TIMEOUT}s. "
                "Try a shorter instruction or a smaller file."
            ),
        }
    except Exception as e:
        logging.error(f"generate_file_edit() unexpected error: {e}")
        return {
            "success": False,
            "extracted_content": None,
            "raw_response": None,
            "error": f"Unexpected error: {str(e)}",
        }
    