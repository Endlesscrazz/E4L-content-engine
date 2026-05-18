"""
Researcher mini-agent for the E4L content engine.

Runs a Haiku 4.5 loop (max 5 actions, $0.10 cap) to find external trend
context that is semantically relevant to the E4L source corpus. Corpus-
relevance judgment is deterministic (cosine similarity) — not delegated
to the LLM. The LLM only decides what to search for and when to stop.

Design split:
  Haiku 4.5  — query formulation, result selection, loop termination
  Python     — action cap, cost cap, consecutive-drop tracking,
               reformulation hints, ResearchFinding construction

Reformulation logic: if two consecutive score_relevance calls return "drop",
Python injects a hint into the next tool_result telling Haiku to change
search direction. If drops continue after reformulation, the loop terminates
early and returns no_context_reason="no trend context available". This keeps
the LLM cheap while making the "give up gracefully" path deterministic.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from app.clients import AnthropicClient, BraveClient, GeminiEmbedder
from app.models import AgentTelemetry, ContentBrief, ResearchFinding, ResearchResult
from app.observability import write_agent_call

logger = logging.getLogger(__name__)

RESEARCHER_MODEL = "claude-haiku-4-5-20251001"
MAX_ACTIONS = 5
MAX_COST_USD = 0.10

# Cap snippet length stored in findings to keep Orchestrator context budget sane.
_SNIPPET_MAX_CHARS = 600
# Cap text sent to Gemini embed — stays within free-tier rate limits.
_GEMINI_INPUT_MAX_CHARS = 4000

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "researcher_system.txt"
_SYSTEM_PROMPT: str | None = None


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text()
    return _SYSTEM_PROMPT


# ─── Tool schemas ─────────────────────────────────────────────────────────────

RESEARCHER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_web",
        "description": (
            "Search the web for recent information on a topic. "
            "Returns titles, URLs, and descriptions of up to 5 results. "
            "Start here to survey what's available before reading or scoring."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific, targeted search query.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": (
            "Fetch and extract plain text from a URL. "
            "Use when a search description is too short to judge relevance. "
            "Costs one action — only use when it will change your scoring decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
                "title": {"type": "string", "description": "Page title for reference."},
            },
            "required": ["url", "title"],
        },
    },
    {
        "name": "score_relevance_to_corpus",
        "description": (
            "Score how relevant a text snippet is to the E4L source corpus. "
            "Returns score (0–1) and label: keep (≥0.55), weak (0.40–0.55), drop (<0.40). "
            "Always call this before deciding to keep a result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to score — description or extracted page content.",
                },
                "title": {"type": "string", "description": "Page title for reference."},
                "url": {"type": "string", "description": "Source URL for reference."},
            },
            "required": ["text", "title", "url"],
        },
    },
]


# ─── Tool execution ───────────────────────────────────────────────────────────

def _exec_search_web(inputs: dict[str, Any], brave: BraveClient) -> dict[str, Any]:
    results = brave.search(inputs["query"], count=5)
    if not results:
        return {"results": [], "message": "No results found. Try a different query."}
    return {
        "results": [
            {"title": r.title, "url": r.url, "description": r.description}
            for r in results
        ]
    }


def _exec_read_url(inputs: dict[str, Any], brave: BraveClient) -> dict[str, Any]:
    text = brave.read_url(inputs["url"])
    if not text:
        return {"text": "", "message": "Could not extract content from this URL."}
    return {"text": text[:8000]}


def _exec_score_relevance(
    inputs: dict[str, Any],
    gemini: GeminiEmbedder,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """
    Embed text via Gemini, find nearest corpus vector via vec0 MATCH, convert
    L2 distance to cosine similarity.

    text-embedding-004 outputs unit-norm vectors. For unit-norm vectors the
    L2 distance d and cosine similarity relate exactly as:
      cosine = 1 - d² / 2
    This is exact (not an approximation) because |a|=|b|=1 by construction.
    If the embedding model changes, verify unit-norm guarantee before using
    this formula — it silently degrades to an approximation for non-unit vectors.
    """
    text = inputs["text"][:_GEMINI_INPUT_MAX_CHARS]
    try:
        query_vec = gemini.embed_query(text)
    except Exception as exc:
        logger.warning("Gemini embed failed during relevance scoring: %s", exc)
        return {"score": 0.0, "label": "drop", "error": "embed_failed"}

    row = conn.execute(
        """
        SELECT distance FROM vec_embeddings
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 1
        """,
        (json.dumps(query_vec),),
    ).fetchone()

    if row is None:
        return {"score": 0.0, "label": "drop", "error": "empty_corpus"}

    d = row["distance"]
    cosine = max(0.0, 1.0 - (d * d) / 2.0)

    if cosine >= 0.55:
        label = "keep"
    elif cosine >= 0.40:
        label = "weak"
    else:
        label = "drop"

    return {"score": round(cosine, 4), "label": label}


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_researcher(
    brief: ContentBrief,
    anthropic_client: AnthropicClient,
    gemini_embedder: GeminiEmbedder,
    brave_client: BraveClient,
    db_conn: sqlite3.Connection,
    trace_id: str,
    obs_conn: sqlite3.Connection | None = None,
) -> ResearchResult:
    """
    Haiku 4.5 agentic loop. Returns ResearchResult with scored web findings.
    findings=[] + no_context_reason set when nothing clears the relevance bar.

    Cap enforcement is in Python — the LLM cannot talk its way past the budget.
    Reformulation hint is injected into tool_result content so Haiku sees it
    naturally in its next turn without a special out-of-band channel.
    """
    topic = brief.topic_focus or brief.goal
    products = ", ".join(brief.product_focus) if brief.product_focus else "general E4L themes"

    initial_message = (
        f"Research external trend context for this content brief.\n"
        f"Goal: {brief.goal}\n"
        f"Audience: {brief.audience.value}\n"
        f"Topic focus: {topic}\n"
        f"Product focus: {products}\n\n"
        "Find recent web content (news, studies, commentary) that intersects "
        "with E4L's themes. Score each promising result before keeping it."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_message}]
    actions_used = 0
    cost_usd = 0.0
    tokens_in_total = 0
    tokens_out_total = 0
    findings: list[ResearchFinding] = []
    consecutive_drops = 0
    reformulated = False
    _loop_start_ms = int(__import__("time").time() * 1000)

    system_prompt = _load_system_prompt()

    while actions_used < MAX_ACTIONS and cost_usd < MAX_COST_USD:
        response, turn_cost = anthropic_client.create(
            model=RESEARCHER_MODEL,
            messages=messages,
            system=system_prompt,
            tools=RESEARCHER_TOOLS,
            max_tokens=1024,
        )
        cost_usd += turn_cost
        tokens_in_total += response.usage.input_tokens
        tokens_out_total += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            logger.warning(
                "[%s] Researcher unexpected stop_reason: %s", trace_id, response.stop_reason
            )
            break

        tool_results: list[dict[str, Any]] = []

        for block in response.content:
            if not hasattr(block, "type") or block.type != "tool_use":
                continue

            actions_used += 1
            name = block.name
            inputs = block.input

            if name == "search_web":
                result = _exec_search_web(inputs, brave_client)

            elif name == "read_url":
                result = _exec_read_url(inputs, brave_client)

            elif name == "score_relevance_to_corpus":
                result = _exec_score_relevance(inputs, gemini_embedder, db_conn)
                label = result.get("label", "drop")

                if label == "drop":
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0
                    findings.append(
                        ResearchFinding(
                            title=inputs.get("title", ""),
                            url=inputs.get("url", ""),
                            snippet=inputs.get("text", "")[:_SNIPPET_MAX_CHARS],
                            relevance_score=result["score"],
                            relevance_label=label,
                        )
                    )

                if consecutive_drops >= 2:
                    if not reformulated:
                        # Inject hint into tool_result — Haiku reads it next turn.
                        result["hint"] = (
                            "Two consecutive results scored below the relevance threshold. "
                            "Abandon this search direction. Try a different angle: "
                            "a synonym, adjacent mechanism, or related E4L concept."
                        )
                        reformulated = True
                        consecutive_drops = 0
                    else:
                        # Already reformulated — still dropping. Return what we have.
                        logger.info(
                            "[%s] Researcher giving up after reformulation: still below threshold",
                            trace_id,
                        )
                        _write_researcher_obs(obs_conn, trace_id, tokens_in_total, tokens_out_total, cost_usd, actions_used, _loop_start_ms)
                        return ResearchResult(
                            findings=findings,
                            no_context_reason="no trend context available",
                            actions_used=actions_used,
                            cost_usd=round(cost_usd, 6),
                        )

            else:
                result = {"error": f"Unknown tool: {name}"}
                logger.warning("[%s] Researcher called unknown tool: %s", trace_id, name)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    _write_researcher_obs(obs_conn, trace_id, tokens_in_total, tokens_out_total, cost_usd, actions_used, _loop_start_ms)

    if not findings:
        return ResearchResult(
            findings=[],
            no_context_reason="no trend context available",
            actions_used=actions_used,
            cost_usd=round(cost_usd, 6),
        )

    return ResearchResult(
        findings=findings,
        actions_used=actions_used,
        cost_usd=round(cost_usd, 6),
    )


def _write_researcher_obs(
    obs_conn: sqlite3.Connection | None,
    trace_id: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    tool_calls: int,
    start_ms: int,
) -> None:
    if obs_conn is None:
        return
    latency_ms = int(__import__("time").time() * 1000) - start_ms
    write_agent_call(
        obs_conn,
        AgentTelemetry(
            trace_id=trace_id,
            agent_name="researcher",
            model=RESEARCHER_MODEL,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost_usd, 6),
            latency_ms=latency_ms,
            tool_calls=tool_calls,
        ),
    )
