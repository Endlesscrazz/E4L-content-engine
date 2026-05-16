"""
Orchestrator agent for the E4L content engine.

Runs a Haiku 4.5 agentic loop that coordinates: research → write → validate → finalize.
Haiku decides the sequence and parallelism; Python enforces all caps and executes
the actual agent calls.

Model: Haiku 4.5 (no temperature — deprecated for Claude 4).
Fallback: Sonnet 4.6 if Haiku planning quality degrades — measure on first live run,
  log result in DECISIONS.md, request permission before switching.

Design split:
  Haiku 4.5   — pipeline planning, tool call sequencing, revision decisions
  Python      — RunBudget enforcement, corpus retrieval, asyncio.gather dispatch,
                state management (drafts, verdicts), DndFlag loading,
                observability writes, SSE event emission

Cap enforcement: RunBudget.is_exhausted() is checked BEFORE dispatching any tool_use.
  If exhausted mid-loop, a budget_exhausted signal is injected and Haiku is prompted
  to finalize. The cap is in Python; the system prompt only tells Haiku what to do
  when it receives the signal — not the actual cap values.

Parallel dispatch: when Haiku calls call_writer for both platforms in one turn,
  asyncio.gather runs them concurrently. Detection: count call_writer blocks.

Revision loop: max MAX_REVISIONS total (both platforms combined). Tracked Python-side.
  When hit, call_writer returns cap_hit and Haiku should finalize.

Corpus retrieval: done once on the first call_writer dispatch via Gemini embedding +
  query_similar. Same chunks reused for both platforms and any revisions.

State management convention: _dispatch_single returns a dict with _-prefixed keys
  for Python-internal state (Draft, ValidatorVerdict, ResearchResult, corpus chunks).
  These are filtered out by _tool_result() before being sent to Anthropic.
  The main loop is the sole place that updates shared mutable state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.clients import AnthropicClient, BraveClient, GeminiEmbedder
from app.corpus_store import get_voice_anchors, query_similar
from app.models import (
    AgentTelemetry,
    ContentBrief,
    ContentPack,
    Draft,
    Platform,
    ResearchResult,
    RunBudget,
    RunRecord,
    SourceChunk,
    ToolCallEvent,
    ValidatorVerdict,
)
from app.observability import (
    write_agent_call,
    write_run_end,
    write_run_start,
    write_tool_event,
)
from app.validator import validate_draft
from app.validator_gates import load_dnd_flags
from agents.researcher import run_researcher
from agents.writer import run_writer

logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = "claude-haiku-4-5-20251001"
# Sonnet 4.6 fallback — swap this constant after measuring and logging results.
# ORCHESTRATOR_MODEL = "claude-sonnet-4-6"

MAX_TURNS = 15
MAX_COST_USD = 0.50
MAX_REVISIONS = 2

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "orchestrator_system.txt"
_SYSTEM_PROMPT: str | None = None
_DND_YAML = Path("corpus/do_not_discuss.yaml")


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


# ─── Orchestrator tool schemas ─────────────────────────────────────────────────

ORCHESTRATOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "call_researcher",
        "description": (
            "Run the Researcher mini-agent to gather external trend context. "
            "Optional — call it before writing if fresh trend data would strengthen the content. "
            "Returns scored web findings or empty when nothing relevant is found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Research goal — what kind of trend context to find.",
                },
                "topic_focus": {
                    "type": "string",
                    "description": "Optional topic to focus the search (e.g. 'mitochondrial health').",
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "call_writer",
        "description": (
            "Run the Writer agent for a platform. "
            "You SHOULD call this for both 'linkedin' and 'email' in the same turn "
            "to generate both in parallel — do not call them in separate turns on first draft. "
            "Pass revision_notes only when revising a rejected draft."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "email"],
                    "description": "The platform to generate content for.",
                },
                "revision_notes": {
                    "type": "string",
                    "description": "Validator feedback to pass to the Writer. Omit on first draft.",
                },
            },
            "required": ["platform"],
        },
    },
    {
        "name": "call_validator",
        "description": (
            "Run the Validator on a completed draft. "
            "Call once per platform after the Writer succeeds. "
            "Returns ship=true if the draft is approved, or revision_notes if rejected."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["linkedin", "email"],
                },
            },
            "required": ["platform"],
        },
    },
    {
        "name": "finalize",
        "description": (
            "End the pipeline and return the content pack. "
            "Call when all platforms are validated and shipped, or immediately when you "
            "receive a budget_exhausted signal, or when you cannot proceed further."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete", "partial", "cap_hit", "refused"],
                    "description": (
                        "complete = all platforms shipped; "
                        "partial = at least one shipped; "
                        "cap_hit = budget exhausted before completion; "
                        "refused = editorial gate hard-failed."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why the pipeline ended.",
                },
            },
            "required": ["status", "reason"],
        },
    },
]


# ─── Main entry point ─────────────────────────────────────────────────────────

async def run_orchestrator(
    brief: ContentBrief,
    anthropic_client: AnthropicClient,
    gemini_embedder: GeminiEmbedder,
    brave_client: BraveClient,
    corpus_conn: sqlite3.Connection,
    obs_conn: sqlite3.Connection | None = None,
    trace_id: str = "",
    on_event: Callable[[str, dict], None] = lambda *_: None,
) -> ContentPack:
    """
    Run the full Orchestrator pipeline. Returns ContentPack with all results.

    obs_conn: open observability sqlite connection. None disables obs writes (tests).
    on_event: SSE stub — S6 replaces this no-op with the actual SSE emitter.
    trace_id: generated at API layer and threaded through. Auto-generated if empty.
    """
    if not trace_id:
        trace_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_auto"

    budget = RunBudget(max_turns=MAX_TURNS, max_cost_usd=MAX_COST_USD)

    if obs_conn is not None:
        write_run_start(
            obs_conn,
            RunRecord(
                trace_id=trace_id,
                brief_json=brief.model_dump_json(),
                status="running",
                start_ts=datetime.now(timezone.utc).isoformat(),
            ),
        )

    on_event("pipeline_start", {"trace_id": trace_id, "goal": brief.goal})

    # ── Shared pipeline state (only modified in the main loop, not in helpers) ──
    drafts: dict[Platform, Draft] = {}
    verdicts: dict[Platform, ValidatorVerdict] = {}
    research: ResearchResult | None = None
    corpus_chunks: list[SourceChunk] | None = None  # fetched once on first call_writer
    voice_anchors: list[SourceChunk] = []
    revisions_used: int = 0
    finalize_status: str = "cap_hit"
    finalize_reason: str = "turn cap reached without finalize call"

    dnd_flags = load_dnd_flags(_DND_YAML) if _DND_YAML.exists() else []

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Generate a LinkedIn post and email for this brief.\n\n"
                f"Goal: {brief.goal}\n"
                f"Audience: {brief.audience.value}\n"
                f"Funnel stage: {brief.funnel_stage.value}\n"
                f"Tone: {brief.tone.value}\n"
                + (f"Topic focus: {brief.topic_focus}\n" if brief.topic_focus else "")
                + (f"Product focus: {', '.join(brief.product_focus)}\n" if brief.product_focus else "")
                + "\nCoordinate the pipeline. Use your tools to research (optional), "
                + "write both platforms in parallel, validate each, and finalize."
            ),
        }
    ]

    system_prompt = _load_system_prompt()

    # ─── Agentic loop ─────────────────────────────────────────────────────────
    while not budget.is_exhausted():
        t0 = datetime.now(timezone.utc)
        response, turn_cost = await asyncio.to_thread(
            anthropic_client.create,
            model=ORCHESTRATOR_MODEL,
            messages=messages,
            system=system_prompt,
            tools=ORCHESTRATOR_TOOLS,
            max_tokens=1024,
        )
        latency_ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        budget.record(turn_cost)

        if obs_conn is not None:
            write_agent_call(
                obs_conn,
                AgentTelemetry(
                    trace_id=trace_id,
                    agent_name="orchestrator",
                    model=ORCHESTRATOR_MODEL,
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    cost_usd=turn_cost,
                    latency_ms=latency_ms,
                    tool_calls=sum(
                        1 for b in response.content if hasattr(b, "type") and b.type == "tool_use"
                    ),
                ),
            )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            logger.warning("[%s] Orchestrator end_turn without finalize", trace_id)
            on_event("pipeline_warning", {"message": "end_turn without finalize"})
            break

        if response.stop_reason != "tool_use":
            logger.warning("[%s] Orchestrator unexpected stop_reason: %s", trace_id, response.stop_reason)
            break

        tool_blocks = [b for b in response.content if hasattr(b, "type") and b.type == "tool_use"]
        tool_results: list[dict[str, Any]] = []
        should_break = False

        # Detect parallel writer dispatch: Haiku calls call_writer for both platforms at once.
        writer_blocks = [b for b in tool_blocks if b.name == "call_writer"]
        other_blocks = [b for b in tool_blocks if b.name != "call_writer"]

        if len(writer_blocks) >= 2:
            # Parallel path — fetch corpus once, gather both Writers
            if corpus_chunks is None:
                corpus_chunks, voice_anchors = await asyncio.to_thread(
                    _fetch_corpus, brief=brief, gemini_embedder=gemini_embedder,
                    corpus_conn=corpus_conn,
                )

            writer_drafts = await _gather_writers(
                writer_blocks=writer_blocks,
                brief=brief,
                corpus_chunks=corpus_chunks,
                voice_anchors=voice_anchors,
                research=research,
                anthropic_client=anthropic_client,
                trace_id=trace_id,
                obs_conn=obs_conn,
                on_event=on_event,
            )
            for block, draft in zip(writer_blocks, writer_drafts):
                platform = Platform(block.input["platform"])
                if draft is not None:
                    drafts[platform] = draft
                    tool_results.append(_tool_result(block.id, {"status": "ok", "platform": platform.value}))
                else:
                    revisions_used += 1
                    tool_results.append(_tool_result(block.id, {
                        "status": "parse_failed",
                        "message": "Draft parse failed. Retry with submit_draft format strictly followed.",
                    }))
        elif len(writer_blocks) == 1:
            # Single writer (e.g. revision for one platform)
            block = writer_blocks[0]
            platform = Platform(block.input["platform"])
            if revisions_used >= MAX_REVISIONS:
                tool_results.append(_tool_result(block.id, {
                    "status": "cap_hit",
                    "message": f"Revision cap ({MAX_REVISIONS}) reached. Call finalize.",
                }))
            else:
                if corpus_chunks is None:
                    corpus_chunks, voice_anchors = await asyncio.to_thread(
                        _fetch_corpus, brief=brief, gemini_embedder=gemini_embedder,
                        corpus_conn=corpus_conn,
                    )
                draft = await run_writer(
                    platform=platform,
                    brief=brief,
                    corpus_chunks=corpus_chunks,
                    voice_anchors=voice_anchors,
                    research=research,
                    revision_notes=block.input.get("revision_notes"),
                    anthropic_client=anthropic_client,
                    trace_id=trace_id,
                    obs_conn=obs_conn,
                    on_event=on_event,
                )
                if draft is not None:
                    drafts[platform] = draft
                    tool_results.append(_tool_result(block.id, {"status": "ok", "platform": platform.value}))
                else:
                    revisions_used += 1
                    tool_results.append(_tool_result(block.id, {
                        "status": "parse_failed",
                        "message": "Draft parse failed. Retry with submit_draft format.",
                    }))

        # Dispatch non-writer tools sequentially
        for block in other_blocks:
            result_dict, fin, fs, fr = await _dispatch_tool(
                block=block,
                brief=brief,
                drafts=drafts,
                verdicts=verdicts,
                research=research,
                corpus_chunks=corpus_chunks,
                voice_anchors=voice_anchors,
                dnd_flags=dnd_flags,
                revisions_used=revisions_used,
                anthropic_client=anthropic_client,
                gemini_embedder=gemini_embedder,
                brave_client=brave_client,
                corpus_conn=corpus_conn,
                trace_id=trace_id,
                obs_conn=obs_conn,
                on_event=on_event,
            )
            # Extract Python-internal state (prefixed with _)
            if block.name == "call_researcher":
                _r = result_dict.pop("_research_result", None)
                if _r is not None:
                    research = _r
            elif block.name == "call_validator":
                _v = result_dict.pop("_verdict", None)
                if _v is not None:
                    verdicts[Platform(block.input["platform"])] = _v

            tool_results.append(_tool_result(block.id, result_dict))

            if fin:
                finalize_status, finalize_reason = fs, fr
                should_break = True
                break

        # Inject budget_exhausted signal if cap hit mid-loop
        if budget.is_exhausted() and not should_break:
            on_event("budget_exhausted", {
                "turns_used": budget.turns_used,
                "cost_usd": budget.cost_usd_spent,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": "budget_signal",
                "content": json.dumps({
                    "type": "budget_exhausted",
                    "message": "Budget cap reached. Call finalize with status='cap_hit' immediately.",
                    "turns_used": budget.turns_used,
                    "cost_usd_spent": round(budget.cost_usd_spent, 4),
                }),
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if should_break:
            break

    # ─── Assemble ContentPack ─────────────────────────────────────────────────
    shipped = {p for p, v in verdicts.items() if v.ship}
    if len(shipped) == 2:
        finalize_status = "complete"
        finalize_reason = "all platforms shipped"
    elif len(shipped) == 1 and finalize_status != "refused":
        finalize_status = "partial"
        finalize_reason = f"only {next(iter(shipped)).value} shipped"

    pack = ContentPack(
        trace_id=trace_id,
        brief=brief,
        linkedin_draft=drafts.get(Platform.LINKEDIN),
        email_draft=drafts.get(Platform.EMAIL),
        linkedin_verdict=verdicts.get(Platform.LINKEDIN),
        email_verdict=verdicts.get(Platform.EMAIL),
        revisions_used=revisions_used,
        status=finalize_status,
        budget=budget,
        research=research,
    )

    if obs_conn is not None:
        write_run_end(
            obs_conn,
            trace_id=trace_id,
            status=finalize_status,
            total_cost_usd=budget.cost_usd_spent,
            turns_used=budget.turns_used,
        )

    on_event("pipeline_done", {
        "trace_id": trace_id,
        "status": finalize_status,
        "turns_used": budget.turns_used,
        "cost_usd": round(budget.cost_usd_spent, 4),
        "shipped_platforms": [p.value for p in shipped],
    })

    logger.info(
        "[%s] Orchestrator done: status=%s turns=%d cost=$%.4f",
        trace_id, finalize_status, budget.turns_used, budget.cost_usd_spent,
    )
    return pack


# ─── Tool dispatcher ──────────────────────────────────────────────────────────

async def _dispatch_tool(
    *,
    block: Any,
    brief: ContentBrief,
    drafts: dict[Platform, Draft],
    verdicts: dict[Platform, ValidatorVerdict],
    research: ResearchResult | None,
    corpus_chunks: list[SourceChunk] | None,
    voice_anchors: list[SourceChunk],
    dnd_flags: list,
    revisions_used: int,
    anthropic_client: AnthropicClient,
    gemini_embedder: GeminiEmbedder,
    brave_client: BraveClient,
    corpus_conn: sqlite3.Connection,
    trace_id: str,
    obs_conn: Any | None,
    on_event: Callable[[str, dict], None],
) -> tuple[dict[str, Any], bool, str, str]:
    """
    Dispatch a single non-writer tool_use block.
    Returns (result_dict, finalized, finalize_status, finalize_reason).

    result_dict keys starting with _ are Python-internal state; the main loop
    extracts them and strips them before _tool_result() serializes for Anthropic.
    """
    name = block.name
    inputs = block.input

    if obs_conn is not None:
        write_tool_event(
            obs_conn,
            ToolCallEvent(
                trace_id=trace_id,
                agent_name="orchestrator",
                tool_name=name,
                input_json=json.dumps(inputs)[:2000],
                output_json="pending",
                ts=datetime.now(timezone.utc).isoformat(),
            ),
        )

    # ── call_researcher ────────────────────────────────────────────────────────
    if name == "call_researcher":
        on_event("researcher_start", {"trace_id": trace_id})
        try:
            result_obj = await asyncio.to_thread(
                run_researcher,
                brief=brief,
                anthropic_client=anthropic_client,
                gemini_embedder=gemini_embedder,
                brave_client=brave_client,
                db_conn=corpus_conn,
                trace_id=trace_id,
            )
            on_event("researcher_done", {
                "findings": len(result_obj.findings),
                "cost_usd": result_obj.cost_usd,
            })
            return {
                "_research_result": result_obj,  # extracted by main loop; not sent to Haiku
                "findings_count": len(result_obj.findings),
                "has_context": len(result_obj.findings) > 0,
            }, False, "", ""
        except Exception as exc:
            logger.error("[%s] Researcher error: %s", trace_id, exc)
            return {"status": "error", "message": str(exc)}, False, "", ""

    # ── call_validator ─────────────────────────────────────────────────────────
    elif name == "call_validator":
        platform = Platform(inputs["platform"])
        if platform not in drafts:
            return {
                "status": "no_draft",
                "message": f"No draft found for {platform.value}. Call call_writer first.",
            }, False, "", ""

        draft = drafts[platform]
        on_event("validator_start", {"platform": platform.value, "trace_id": trace_id})
        try:
            chunks_dict = {c.chunk_id: c for c in (corpus_chunks or [])}
            verdict = await asyncio.to_thread(
                validate_draft,
                draft=draft,
                corpus_chunks=chunks_dict,
                brief=brief,
                dnd_flags=dnd_flags,
                voice_anchors=voice_anchors,
                client=anthropic_client,
            )

            on_event("validator_done", {
                "platform": platform.value,
                "ship": verdict.ship,
                "revision_notes": verdict.revision_notes,
            })

            result: dict[str, Any] = {
                "_verdict": verdict,  # extracted by main loop; not sent to Haiku
                "ship": verdict.ship,
                "platform": platform.value,
            }
            if not verdict.ship and verdict.revision_notes:
                result["revision_notes"] = verdict.revision_notes
                result["message"] = "Draft rejected. Pass revision_notes to call_writer to revise."
            return result, False, "", ""

        except Exception as exc:
            logger.error("[%s] Validator error on %s: %s", trace_id, platform.value, exc)
            return {"status": "error", "message": str(exc)}, False, "", ""

    # ── finalize ───────────────────────────────────────────────────────────────
    elif name == "finalize":
        status = inputs.get("status", "partial")
        reason = inputs.get("reason", "")
        on_event("pipeline_finalizing", {"status": status, "reason": reason})
        return {"status": "finalized", "reason": reason}, True, status, reason

    else:
        logger.warning("[%s] Unknown tool: %s", trace_id, name)
        return {"error": f"Unknown tool: {name}"}, False, "", ""


# ─── Parallel writer gather ───────────────────────────────────────────────────

async def _gather_writers(
    *,
    writer_blocks: list[Any],
    brief: ContentBrief,
    corpus_chunks: list[SourceChunk],
    voice_anchors: list[SourceChunk],
    research: ResearchResult | None,
    anthropic_client: AnthropicClient,
    trace_id: str,
    obs_conn: Any | None,
    on_event: Callable[[str, dict], None],
) -> list[Draft | None]:
    """Run Writer calls concurrently via asyncio.gather. Returns drafts in block order."""

    async def _one(block: Any) -> Draft | None:
        return await run_writer(
            platform=Platform(block.input["platform"]),
            brief=brief,
            corpus_chunks=corpus_chunks,
            voice_anchors=voice_anchors,
            research=research,
            revision_notes=block.input.get("revision_notes"),
            anthropic_client=anthropic_client,
            trace_id=trace_id,
            obs_conn=obs_conn,
            on_event=on_event,
        )

    return list(await asyncio.gather(*[_one(b) for b in writer_blocks]))


# ─── Corpus retrieval ─────────────────────────────────────────────────────────

def _fetch_corpus(
    *,
    brief: ContentBrief,
    gemini_embedder: GeminiEmbedder,
    corpus_conn: sqlite3.Connection,
) -> tuple[list[SourceChunk], list[SourceChunk]]:
    """Embed the brief goal and retrieve top-k relevant chunks + all voice anchors."""
    query_text = " ".join(filter(None, [brief.goal, brief.topic_focus, *brief.product_focus]))
    try:
        query_vec = gemini_embedder.embed_query(query_text)
        chunks = query_similar(
            conn=corpus_conn,
            query_embedding=query_vec,
            top_k=8,
            audience_filter=brief.audience.value if brief.audience else None,
        )
    except Exception as exc:
        logger.warning("Corpus retrieval failed: %s — proceeding with empty chunks", exc)
        chunks = []

    try:
        anchors = get_voice_anchors(corpus_conn)
    except Exception as exc:
        logger.warning("Voice anchor retrieval failed: %s", exc)
        anchors = []

    return chunks, anchors


# ─── Anthropic message helpers ────────────────────────────────────────────────

def _tool_result(tool_use_id: str, content: dict[str, Any]) -> dict[str, Any]:
    """
    Build a tool_result block for the Anthropic messages API.
    Keys starting with _ are Python-internal and are stripped before JSON serialization.
    Non-JSON-serializable values are coerced to their string representation.
    """
    clean: dict[str, Any] = {}
    for k, v in content.items():
        if k.startswith("_"):
            continue  # internal Python state, not for Haiku
        if isinstance(v, (str, int, float, bool, type(None))):
            clean[k] = v
        elif isinstance(v, (list, dict)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(clean),
    }
