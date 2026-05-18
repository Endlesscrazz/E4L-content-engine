"""
Security utilities: prompt injection detection and text sanitization.

Three threat surfaces addressed:
  1. User input (ContentBrief fields) — detect injection, reject with 422
  2. External content (Brave search results, fetched URLs) — strip structural
     characters that could be misread as prompt instructions
  3. Writer output passed to downstream agents — light scan before Validator sees it

Design choice: detect-and-reject on user input rather than silently strip.
Stripping can mask an attack while still leaking partial content into the prompt;
rejection surfaces the problem and gives the user a clear error.
"""

from __future__ import annotations

import re
import unicodedata

# ─── Injection detection ──────────────────────────────────────────────────────

# Patterns targeting common jailbreak / instruction-override techniques.
# Not exhaustive — layered defence means the editorial pre-gate and validator
# are the last line; this is the first line at the API boundary.
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(?:previous|prior|the\s+above)\s+instructions?",
    r"disregard\s+(all\s+)?(?:previous|prior|the\s+above)\s+instructions?",
    r"forget\s+(?:previous|prior|all|everything)",
    r"you\s+are\s+now\s+(?:a|an|the)\s+\w",
    r"act\s+as\s+(?:if\s+you\s+(?:are|were)|a|an)\s+",
    r"new\s+(?:system\s+)?instructions?\s*:",
    r"<\s*/?system\s*>",
    r"<\s*/?instructions?\s*>",
    r"\[system\]",
    r"#{1,6}\s*system\b",
    r"#{1,6}\s*instructions?\b",
    r"\bdan\s+mode\b",
    r"\bjailbreak\b",
    r"prompt\s+injection",
    r"do\s+anything\s+now",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


def contains_injection(text: str) -> bool:
    """True if text contains a known prompt injection pattern."""
    for pattern in _COMPILED:
        if pattern.search(text):
            return True
    return False


# ─── Sanitization helpers ─────────────────────────────────────────────────────

def sanitize_user_text(text: str, max_length: int) -> str:
    """
    Light sanitization for user-supplied text. Removes null bytes and
    C0/C1 control characters (except newline and tab), collapses runs of
    whitespace, truncates. Injection detection is handled separately via
    contains_injection() — we reject rather than silently mangle.
    """
    # Strip control chars except \n and \t
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t"
    )
    # Collapse repeated spaces/tabs (preserve single newlines for readability)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()[:max_length]


def sanitize_external_text(text: str, max_length: int = 800) -> str:
    """
    Sanitization for text from external sources (Brave results, fetched URLs).
    More aggressive: removes angle-bracket markup and role-header patterns
    that could be interpreted as prompt instructions by downstream LLM calls.

    Why strip rather than reject: we can't refuse a Brave result; we must
    include it or lose research context. Stripping structural characters
    is safer than passing raw HTML/markdown that could contain injected directives.
    """
    if not text:
        return ""
    # Control chars
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t"
    )
    # HTML/XML-style tags (up to 200 chars to avoid false positives on < in math)
    text = re.sub(r"<[^>]{0,200}>", " ", text)
    # Markdown role headers that could be mistaken for chat turn markers
    text = re.sub(r"^#{1,6}\s*(system|user|assistant|instructions?)\b.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()[:max_length]


def scan_agent_output(text: str) -> bool:
    """
    Lightweight scan of LLM output before passing to a downstream agent.
    Returns True if the output looks clean, False if injection patterns
    are detected (caller should log and skip rather than propagate).

    Rationale: a compromised or hallucinating writer could embed directives
    in draft body text. The validator is the authoritative judge, but this
    scan catches gross cases before they enter the validator's context.
    """
    return not contains_injection(text)
