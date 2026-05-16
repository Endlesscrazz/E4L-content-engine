"""
Deterministic validator gates — zero LLM cost, fully unit-testable.

Pipeline order (short-circuits on first failure before any API call):
  Layer 0  check_editorial      — lexicon/regex pre-gate
  Layer 1  check_citations      — normalized substring resolution
  Layer 4  check_do_not_discuss — citation join + draft-body scan

Layers 2+3 (Opus LLM judge) run only after all three pass; implemented in S3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import yaml

from app.models import (
    CitationCheckResult,
    ContentBrief,
    DoNotDiscussCheckResult,
    Draft,
    EditorialCheckResult,
    SourceChunk,
)


# ─── Layer 0: Editorial pre-gate ──────────────────────────────────────────────
#
# Narrow patterns: fire only on clear medical overreach, not hedged copy.
# "treat yourself" is intentionally NOT flagged — only treat + disease noun.
# False-positive tolerance is low by design: edge cases route to the LLM judge.
#
# Each entry: (regex_pattern, label_for_triggered_phrase)

EDITORIAL_PATTERNS: list[tuple[str, str]] = [
    # Cure claims — always overreach in wellness marketing
    (r"\bcure[sd]?\b|\bcuring\b", "cure claim"),
    # Diagnose — BWS doc: "does not make any sort of medical diagnosis"
    (r"\bdiagnose[sd]?\b|\bdiagnosis\b", "diagnose claim"),
    # Treat + disease noun (narrow: excludes "treat yourself", "treat symptoms naturally")
    (
        r"\btreat(s|ed|ing)?\b.{0,60}"
        r"\b(chronic\s+\w+|disease|condition|illness|disorder|cancer"
        r"|diabetes|depression|anxiety|autoimmune)\b",
        "treat-disease claim",
    ),
    # Stop / replace medication
    (
        r"\b(stop|quit|discontinue|replace|wean\s+off)\b.{0,60}"
        r"\b(medication[s]?|medicine[s]?|drug[s]?|prescription[s]?|pharmaceutical[s]?)\b",
        "stop/replace medication claim",
    ),
    # Anti-vaccine
    (r"\banti.?vacc(ine[sd]?|ination[s]?)?\b", "anti-vaccine claim"),
    (
        r"\bvaccine[s]?\b.{0,40}\b(cause[sd]?|harmful|dangerous|don.t work|ineffective)\b",
        "anti-vaccine claim",
    ),
    # FDA-approved — E4L devices are not FDA-cleared for medical claims
    (r"\bFDA.?approved\b", "FDA-approved claim"),
    # Absolute efficacy
    (r"\b100\s*%\s+(effective|success|guaranteed|cure[sd]?|heal[s]?)\b", "absolute efficacy claim"),
    (r"\bguaranteed\s+(to\s+)?(work|cure|heal|treat|fix)\b", "absolute efficacy claim"),
    (r"\balways\s+(work[s]?|cure[sd]?|heal[s]?|eliminate[s]?)\b", "absolute efficacy claim"),
    (r"\bnever\s+fail[s]?\b", "absolute efficacy claim"),
    # Severity / chronicity reversal
    (
        r"\bpermanently\s+(cure[sd]?|heal[sd]?|fix(ed)?|eliminate[sd]?)\b",
        "severity/chronicity reversal claim",
    ),
    (
        r"\b(reverse[sd]?|eliminate[sd]?)\b.{0,50}"
        r"\b(chronic|disease|condition|illness|disorder)\b",
        "severity/chronicity reversal claim",
    ),
]


def check_editorial(draft: Draft) -> EditorialCheckResult:
    """Layer 0 — editorial pre-gate. No API cost; runs first on every draft."""
    full_text = " ".join(filter(None, [draft.subject, draft.body]))
    for pattern, label in EDITORIAL_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            return EditorialCheckResult(
                passed=False,
                triggered_phrase=f"{label}: '{match.group(0)}'",
                message=f"Draft fails editorial pre-gate — {label}.",
            )
    return EditorialCheckResult(passed=True)


# ─── Layer 1: Citation resolution ─────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Smart quotes → straight quotes; strip whitespace.
    Docx exports U+2018/2019/201C/201D — normalize so substring match is encoding-agnostic.
    Case is preserved: citation substrings should be exact quotes, not case-folded."""
    return (
        text
        .replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .strip()
    )


def check_citations(
    draft: Draft,
    corpus_chunks: dict[str, SourceChunk],
) -> CitationCheckResult:
    """Layer 1 — citation resolution gate.
    Every Claim with chunk_id + cited_substring must have the substring in that chunk's content.
    Type-C claims (chunk_id=None) are intentionally skipped — they cite no source."""
    failed: list[str] = []
    for claim in draft.claims:
        if claim.chunk_id is None or claim.cited_substring is None:
            continue
        chunk = corpus_chunks.get(claim.chunk_id)
        if chunk is None:
            failed.append(claim.text)
            continue
        if _normalize(claim.cited_substring) not in _normalize(chunk.content):
            failed.append(claim.text)

    if failed:
        return CitationCheckResult(
            passed=False,
            failed_claims=failed,
            message=f"Citation resolution failed for {len(failed)} claim(s).",
        )
    return CitationCheckResult(passed=True)


# ─── Layer 4 (runs 3rd): do_not_discuss gate ──────────────────────────────────

@dataclass
class DndFlag:
    flag: str
    mode: str               # "never_in_generated_content" | "do_not_volunteer"
    trigger_regex: str
    description: str = ""
    annotation_source: str = ""
    chunk_content_patterns: list[str] = dc_field(default_factory=list)


def load_dnd_flags(yaml_path: Path) -> list[DndFlag]:
    """Load mode-aware flags from do_not_discuss.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [
        DndFlag(
            flag=item["flag"],
            mode=item["mode"],
            trigger_regex=item["trigger_regex"],
            description=item.get("description", ""),
            annotation_source=item.get("annotation_source", ""),
            chunk_content_patterns=item.get("chunk_content_patterns", []),
        )
        for item in data.get("do_not_discuss_flags", [])
    ]


def _find_flag_for_chunk(chunk: SourceChunk, flags: list[DndFlag]) -> DndFlag | None:
    """Identify which DndFlag a flagged chunk belongs to by matching trigger_regex to chunk content."""
    for flag in flags:
        if re.search(flag.trigger_regex, chunk.content, re.IGNORECASE):
            return flag
    return None


def check_do_not_discuss(
    draft: Draft,
    cited_chunks: list[SourceChunk],
    flags: list[DndFlag],
    brief: ContentBrief | None = None,
) -> DoNotDiscussCheckResult:
    """Layer 4 (runs 3rd) — mode-aware do_not_discuss gate.

    Two independent passes:
    Pass 1 (citation join): Did the Writer cite a flagged chunk?
    Pass 2 (body scan): Does the draft text match a flag's trigger_regex?
      — Catches parametric-knowledge evasion: the model may 'know' Peter Fraser
        died without ever citing the flagged chunk.

    do_not_volunteer: allowed if the ContentBrief explicitly mentions the restricted
    topic (user asked for it directly). brief=None → treated as not asked (safe default)."""
    brief_text = ""
    if brief is not None:
        brief_text = " ".join(filter(None, [brief.goal, brief.topic_focus]))

    # Pass 1: citation join
    for chunk in cited_chunks:
        if not chunk.do_not_discuss or chunk.do_not_discuss_mode is None:
            continue

        if chunk.do_not_discuss_mode == "never_in_generated_content":
            matched = _find_flag_for_chunk(chunk, flags)
            return DoNotDiscussCheckResult(
                passed=False,
                triggered_flag=matched.flag if matched else "unknown",
                triggered_mode="never_in_generated_content",
                message=f"Draft cites a chunk flagged never_in_generated_content: {chunk.chunk_id}",
            )

        elif chunk.do_not_discuss_mode == "do_not_volunteer":
            matched = _find_flag_for_chunk(chunk, flags)
            if matched is None:
                continue
            if not (brief_text and re.search(matched.trigger_regex, brief_text, re.IGNORECASE)):
                return DoNotDiscussCheckResult(
                    passed=False,
                    triggered_flag=matched.flag,
                    triggered_mode="do_not_volunteer",
                    message=(
                        f"Draft cites a do_not_volunteer chunk without explicit user request: "
                        f"{chunk.chunk_id}"
                    ),
                )

    # Pass 2: regex body scan
    full_text = " ".join(filter(None, [draft.subject, draft.body]))
    for flag in flags:
        if not re.search(flag.trigger_regex, full_text, re.IGNORECASE):
            continue

        if flag.mode == "never_in_generated_content":
            return DoNotDiscussCheckResult(
                passed=False,
                triggered_flag=flag.flag,
                triggered_mode="never_in_generated_content",
                message=f"Draft text matches never_in_generated_content flag: {flag.flag}",
            )

        elif flag.mode == "do_not_volunteer":
            if not (brief_text and re.search(flag.trigger_regex, brief_text, re.IGNORECASE)):
                return DoNotDiscussCheckResult(
                    passed=False,
                    triggered_flag=flag.flag,
                    triggered_mode="do_not_volunteer",
                    message=f"Draft mentions restricted topic without explicit user request: {flag.flag}",
                )

    return DoNotDiscussCheckResult(passed=True)
