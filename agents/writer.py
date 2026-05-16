"""
Writer agents for the E4L content engine.

One function: run_writer() covers both LinkedIn and Email platforms.
Platform is determined by the `platform` argument; the appropriate prompt
file is loaded for each.

Model: Sonnet 4.6 (no temperature — deprecated for Claude 4).
Tool use: tool_choice forces submit_draft — no free-text response path.
Coercion: _coerce_list / _coerce_claim guard against Sonnet's occasional
  tendency to stringify arrays and nested objects (~15% rate, observed in S3).

Design split:
  Sonnet 4.6  — prose generation, claim extraction, taxonomy labelling
  Python      — forced tool call, coercion guardrails, parse failure retry,
                cost tracking, observability writes

Revision path: if parse fails or Validator rejects, caller (Orchestrator) passes
revision_notes back. Writer receives them as context in the user message. Parse
failure counts against the Orchestrator's revision cap.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.clients import AnthropicClient
from app.models import (
    AgentTelemetry,
    Claim,
    ClaimTaxonomy,
    ContentBrief,
    Draft,
    Platform,
    ResearchResult,
    SourceChunk,
    ToolCallEvent,
)

logger = logging.getLogger(__name__)

WRITER_MODEL = "claude-sonnet-4-6"
# Sonnet 4.6: no temperature — deprecated for all Claude 4 models.

_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt(filename: str) -> str:
    if filename not in _PROMPT_CACHE:
        _PROMPT_CACHE[filename] = (_PROMPT_DIR / filename).read_text(encoding="utf-8")
    return _PROMPT_CACHE[filename]


# ─── submit_draft tool schema ──────────────────────────────────────────────────
#
# tool_choice forces this call — no free-text response path.
# Claim items are typed objects; _coerce_claim() handles Sonnet's occasional
# stringification of nested dicts.

SUBMIT_DRAFT_TOOL: dict[str, Any] = {
    "name": "submit_draft",
    "description": (
        "Submit the completed draft with all cited claims annotated. "
        "Call exactly once when the draft is ready. "
        "Every health or product claim must be labelled with its taxonomy type."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "body": {
                "type": "string",
                "description": "The full draft body text.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line. Required when platform is email.",
            },
            "claims": {
                "type": "array",
                "description": "Every health or product claim in the draft, annotated.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The exact claim text as it appears in the draft.",
                        },
                        "chunk_id": {
                            "type": "string",
                            "description": "The corpus chunk_id this claim cites. Omit for Type-C.",
                        },
                        "cited_substring": {
                            "type": "string",
                            "description": "The exact substring from the cited chunk that supports this claim.",
                        },
                        "taxonomy": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"],
                            "description": (
                                "A=direct paraphrase (must cite), "
                                "B=inferred from source (must cite + flag), "
                                "C=general knowledge (no citation needed), "
                                "D=novel claim not in source (HARD FAIL — do not use)."
                            ),
                        },
                        "certainty_inflation": {
                            "type": "boolean",
                            "description": "True if the claim's certainty exceeds what the source supports.",
                        },
                    },
                    "required": ["text", "taxonomy", "certainty_inflation"],
                },
            },
            "cited_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "All chunk_ids cited anywhere in the draft.",
            },
        },
        "required": ["body", "claims", "cited_chunk_ids"],
    },
}


# ─── Coercion guardrails ───────────────────────────────────────────────────────
# Sonnet 4.6 occasionally returns arrays or nested objects as JSON-encoded strings.
# These functions normalize the output before Draft construction.
# Observed in S3 on certainty_inflation_issues (~15% of calls). Writer output
# is more complex (nested Claim objects), so both coercions are applied.

def _coerce_list(val: Any) -> list:
    """Return val as a list; parse if Sonnet returned a JSON-encoded string."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [val]
    return []


def _coerce_claim(val: Any) -> dict:
    """Return val as a dict; parse if Sonnet returned a JSON-encoded string."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return val  # return as-is; _parse_draft will log and skip


def _parse_draft(tool_input: dict[str, Any], platform: Platform) -> Draft | None:
    """
    Map submit_draft tool arguments to a Draft.
    Returns None on parse failure so the caller can request a revision.

    Coercion is applied to the claims array and each individual Claim object
    before construction to handle Sonnet's occasional stringification of
    nested structures.
    """
    try:
        raw_claims = _coerce_list(tool_input.get("claims", []))
        claims: list[Claim] = []
        for item in raw_claims:
            item = _coerce_claim(item)
            if not isinstance(item, dict):
                logger.warning("Writer: skipping non-dict claim item: %r", item)
                continue
            try:
                taxonomy_raw = item.get("taxonomy", "C")
                claims.append(
                    Claim(
                        text=item.get("text", ""),
                        chunk_id=item.get("chunk_id"),
                        cited_substring=item.get("cited_substring"),
                        taxonomy=ClaimTaxonomy(taxonomy_raw),
                        certainty_inflation=bool(item.get("certainty_inflation", False)),
                    )
                )
            except Exception as exc:
                logger.warning("Writer: skipping malformed claim %r: %s", item, exc)

        cited_ids = _coerce_list(tool_input.get("cited_chunk_ids", []))

        return Draft(
            platform=platform,
            subject=tool_input.get("subject"),
            body=tool_input["body"],
            claims=claims,
            cited_chunk_ids=[str(c) for c in cited_ids],
        )
    except Exception as exc:
        logger.warning("Writer: _parse_draft failed: %s", exc)
        return None


# ─── Main entry point ─────────────────────────────────────────────────────────

async def run_writer(
    platform: Platform,
    brief: ContentBrief,
    corpus_chunks: list[SourceChunk],
    voice_anchors: list[SourceChunk],
    research: ResearchResult | None,
    revision_notes: str | None,
    anthropic_client: AnthropicClient,
    trace_id: str,
    obs_conn: Any | None = None,
    on_event: Callable[[str, dict], None] = lambda *_: None,
) -> Draft | None:
    """
    Run the Writer for a single platform. Returns Draft or None on parse failure.

    Corpus chunks, voice anchors, and research are injected by the Orchestrator
    (Python-side retrieval) — not derived from Haiku tool calls.

    obs_conn: open observability sqlite connection. None disables obs writes (tests).
    on_event: SSE stub — S6 wires this to the event stream.

    Single-turn: Writer gets one shot (tool forced). If parse fails, returns None
    so Orchestrator can treat it as a failed draft and request revision.
    """
    prompt_file = (
        "writer_linkedin.txt" if platform == Platform.LINKEDIN else "writer_email.txt"
    )
    system_prompt = _load_prompt(prompt_file)

    user_message = _build_writer_message(
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=research,
        revision_notes=revision_notes,
        platform=platform,
    )

    on_event("writer_start", {"platform": platform.value, "trace_id": trace_id})

    t0 = datetime.now(timezone.utc)
    try:
        response, cost = await asyncio.to_thread(
            anthropic_client.create,
            model=WRITER_MODEL,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=[SUBMIT_DRAFT_TOOL],
            tool_choice={"type": "tool", "name": "submit_draft"},
            max_tokens=2048,
        )
    except Exception as exc:
        logger.error("[%s] Writer (%s) API error: %s", trace_id, platform.value, exc)
        on_event("writer_error", {"platform": platform.value, "error": str(exc)})
        return None

    latency_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)

    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    # Find submit_draft tool block
    tool_block = None
    for block in response.content:
        if hasattr(block, "type") and block.type == "tool_use" and block.name == "submit_draft":
            tool_block = block
            break

    if tool_block is None:
        logger.warning(
            "[%s] Writer (%s) did not call submit_draft. stop_reason=%s",
            trace_id,
            platform.value,
            response.stop_reason,
        )
        on_event("writer_error", {"platform": platform.value, "error": "no submit_draft tool call"})
        _write_agent_obs(obs_conn, trace_id, platform, tokens_in, tokens_out, cost, latency_ms, "no_submit_draft")
        return None

    draft = _parse_draft(tool_block.input, platform)

    if draft is None:
        logger.warning("[%s] Writer (%s) parse failed after coercion", trace_id, platform.value)
        on_event("writer_error", {"platform": platform.value, "error": "parse_failed"})
        _write_agent_obs(obs_conn, trace_id, platform, tokens_in, tokens_out, cost, latency_ms, "parse_failed")
        return None

    draft.model_used = WRITER_MODEL
    draft.tokens_in = tokens_in
    draft.tokens_out = tokens_out
    draft.cost_usd = cost

    # Write observability
    _write_agent_obs(obs_conn, trace_id, platform, tokens_in, tokens_out, cost, latency_ms, None)
    if obs_conn is not None:
        from app.observability import write_tool_event
        write_tool_event(
            obs_conn,
            ToolCallEvent(
                trace_id=trace_id,
                agent_name=f"writer_{platform.value}",
                tool_name="submit_draft",
                input_json=json.dumps(tool_block.input)[:4000],
                output_json=json.dumps({"body_length": len(draft.body), "claims": len(draft.claims)}),
                ts=datetime.now(timezone.utc).isoformat(),
            ),
        )

    on_event("writer_done", {
        "platform": platform.value,
        "trace_id": trace_id,
        "claims": len(draft.claims),
        "cited_chunks": len(draft.cited_chunk_ids),
        "cost_usd": cost,
    })

    return draft


# ─── Message builder ──────────────────────────────────────────────────────────

def _build_writer_message(
    brief: ContentBrief,
    corpus_chunks: list[SourceChunk],
    voice_anchors: list[SourceChunk],
    research: ResearchResult | None,
    revision_notes: str | None,
    platform: Platform,
) -> str:
    parts: list[str] = []

    parts.append("## CONTENT BRIEF")
    parts.append(f"Goal: {brief.goal}")
    parts.append(f"Audience: {brief.audience.value}")
    parts.append(f"Funnel stage: {brief.funnel_stage.value}")
    parts.append(f"Tone: {brief.tone.value}")
    if brief.topic_focus:
        parts.append(f"Topic focus: {brief.topic_focus}")
    if brief.product_focus:
        parts.append(f"Product focus: {', '.join(brief.product_focus)}")
    if brief.format_intent:
        parts.append(f"Format intent: {brief.format_intent}")

    parts.append("\n## SOURCE CORPUS CHUNKS (cite from these only)")
    if corpus_chunks:
        for chunk in corpus_chunks:
            conflict_note = " [CORPUS CONFLICT: numeric values in this chunk may contradict another chunk — do not cite specific numbers]" if chunk.corpus_conflict else ""
            parts.append(f"\n[{chunk.chunk_id}] {chunk.doc_name} ({chunk.doc_type}){conflict_note}")
            parts.append(chunk.content)
    else:
        parts.append("(no corpus chunks provided — use only Type-C general knowledge claims)")

    parts.append("\n## VOICE ANCHOR EXAMPLES (authentic Harry Massey / E4L voice)")
    if voice_anchors:
        for anchor in voice_anchors[:4]:  # cap at 4 anchors to stay context-efficient
            parts.append(f"\n[{anchor.chunk_id}]")
            parts.append(anchor.content)
    else:
        parts.append("(no voice anchors available — infer voice from corpus chunks)")

    if research and research.findings:
        parts.append("\n## EXTERNAL RESEARCH CONTEXT (enrichment only — do not cite as E4L source)")
        for finding in research.findings:
            parts.append(f"\n• {finding.title} ({finding.url})")
            parts.append(f"  {finding.snippet}")

    if revision_notes:
        parts.append("\n## REVISION NOTES FROM VALIDATOR")
        parts.append("The previous draft was rejected. Address these specific issues:")
        parts.append(revision_notes)
        parts.append("Do not repeat the same mistake. Call submit_draft once with the corrected draft.")

    parts.append(f"\nPlatform: {platform.value.upper()}")

    return "\n".join(parts)


# ─── Observability helper ──────────────────────────────────────────────────────

def _write_agent_obs(
    obs_conn: Any | None,
    trace_id: str,
    platform: Platform,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    latency_ms: int,
    error: str | None,
) -> None:
    if obs_conn is None:
        return
    try:
        from app.observability import write_agent_call
        write_agent_call(
            obs_conn,
            AgentTelemetry(
                trace_id=trace_id,
                agent_name=f"writer_{platform.value}",
                model=WRITER_MODEL,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                latency_ms=latency_ms,
                tool_calls=1,
                error=error,
            ),
        )
    except Exception as exc:
        logger.warning("Writer obs write failed: %s", exc)
