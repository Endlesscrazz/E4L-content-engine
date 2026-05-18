"""
FastAPI application — S6/S7 API surface.

Seven endpoints:
  POST /generate                — accepts ContentBrief, starts pipeline in background, returns trace_id
  GET  /stream/{trace_id}       — SSE stream of pipeline events (closes after pipeline_done)
  GET  /result/{trace_id}       — full ContentPack once complete; 202 if running; 404 if not found
  GET  /runs                    — last 20 run summaries from obs.db (includes is_replay, replayed_from)
  GET  /corpus/chunk/{id}       — raw corpus chunk for citation click-through viewer
  GET  /runs/{trace_id}/detail  — full per-agent trace: agent_calls + tool_call_events with input_summary
  GET  /runs/{trace_id}/replay  — re-execute same brief under a new trace_id; returns immediately

SSE design: one asyncio.Queue per trace_id in EventManager.
  on_event callback (passed to run_orchestrator) pushes events to the queue.
  /stream consumes the queue via an async generator. None sentinel closes the stream.
  5-min TTL cleanup prevents queue leak on abandoned streams.

Result store: in-memory dict (trace_id → ContentPack). Sufficient for demo lifetime;
  obs.db is the durable audit trail. /result falls back to 404 on server restart.

Clients (Anthropic, Gemini, Brave) and DB connections are initialised once at startup
  via the FastAPI lifespan and stored on app.state — never instantiated per-request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import time
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agents.orchestrator import run_orchestrator
from app.clients import AnthropicClient, BraveClient, GeminiEmbedder
from app.corpus_store import get_chunk, get_conn
from app.models import ContentBrief, ContentPack
from app.observability import delete_run, init_obs_db, query_run_detail, query_runs

load_dotenv(".env.local", override=True)
load_dotenv()
logger = logging.getLogger(__name__)

_STREAM_TTL_SECONDS = 300  # clean up queues 5 min after pipeline_done

# ─── In-process rate limiter ──────────────────────────────────────────────────
# Simple sliding-window counter per client IP. No external dependency needed.
# SlowAPI's decorator approach corrupts FastAPI's type-annotation introspection,
# causing Pydantic body params to be misidentified as query params.
_RATE_WINDOW_S = 60.0
_RATE_LIMIT = 10  # requests per window
_rate_tracker: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    window_start = now - _RATE_WINDOW_S
    hits = [t for t in _rate_tracker[client_ip] if t > window_start]
    _rate_tracker[client_ip] = hits
    if len(hits) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {_RATE_LIMIT} requests per minute per IP.",
        )
    _rate_tracker[client_ip].append(now)


# ─── Event queue manager ──────────────────────────────────────────────────────

class EventManager:
    """
    One asyncio.Queue per active trace_id.
    push() is called from within the async pipeline task (same event loop).
    close() puts a None sentinel so the SSE generator exits cleanly.
    cleanup() removes the queue after the TTL — prevents unbounded growth.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue] = {}

    def create(self, trace_id: str) -> None:
        self._queues[trace_id] = asyncio.Queue()

    def push(self, trace_id: str, event_type: str, data: dict) -> None:
        q = self._queues.get(trace_id)
        if q is not None:
            q.put_nowait({**data, "event_type": event_type})

    def close(self, trace_id: str) -> None:
        q = self._queues.get(trace_id)
        if q is not None:
            q.put_nowait(None)

    def get_queue(self, trace_id: str) -> asyncio.Queue | None:
        return self._queues.get(trace_id)

    def cleanup(self, trace_id: str) -> None:
        self._queues.pop(trace_id, None)


_event_manager = EventManager()

# In-memory result store — survives for process lifetime, sufficient for demo.
_result_store: dict[str, ContentPack] = {}
_running_traces: set[str] = set()


# ─── Lifespan: initialise shared resources once ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.anthropic = AnthropicClient()
    app.state.gemini = GeminiEmbedder()
    app.state.brave = BraveClient()
    app.state.corpus_conn = get_conn()
    app.state.obs_conn = init_obs_db()
    logger.info("E4L Content Engine ready")
    yield
    app.state.corpus_conn.close()
    app.state.obs_conn.close()
    logger.info("E4L Content Engine shutdown")


app = FastAPI(title="E4L Content Engine", lifespan=lifespan)

_STATIC = Path("static")
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")


# ─── Background pipeline task ─────────────────────────────────────────────────

async def _run_pipeline(
    trace_id: str,
    brief: ContentBrief,
    app_state: Any,
    is_replay: bool = False,
    replayed_from: str | None = None,
) -> None:
    _running_traces.add(trace_id)

    def on_event(event_type: str, data: dict) -> None:
        _event_manager.push(trace_id, event_type, data)

    try:
        pack = await run_orchestrator(
            brief=brief,
            trace_id=trace_id,
            anthropic_client=app_state.anthropic,
            gemini_embedder=app_state.gemini,
            brave_client=app_state.brave,
            corpus_conn=app_state.corpus_conn,
            obs_conn=app_state.obs_conn,
            on_event=on_event,
            is_replay=is_replay,
            replayed_from=replayed_from,
        )
        _result_store[trace_id] = pack
    except Exception as exc:
        logger.exception("[%s] Pipeline error: %s", trace_id, exc)
        _event_manager.push(trace_id, "pipeline_error", {"message": str(exc)})
    finally:
        _running_traces.discard(trace_id)
        _event_manager.close(trace_id)
        asyncio.create_task(_delayed_cleanup(trace_id))


async def _delayed_cleanup(trace_id: str) -> None:
    await asyncio.sleep(_STREAM_TTL_SECONDS)
    _event_manager.cleanup(trace_id)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


@app.post("/generate")
async def generate(request: Request, brief: ContentBrief) -> dict:
    """
    Accept a ContentBrief, start the pipeline as a background task, return trace_id.
    Rate-limited to 10 req/min per IP via _check_rate_limit (sliding window, in-process).
    ContentBrief fields are sanitized + injection-checked in model validators before
    this handler runs; a 422 is returned automatically on validation failure.
    The client should immediately open GET /stream/{trace_id} for live events.
    """
    _check_rate_limit(request.client.host if request.client else "unknown")
    trace_id = str(uuid.uuid4())
    _event_manager.create(trace_id)
    asyncio.create_task(_run_pipeline(trace_id, brief, app.state))
    return {"trace_id": trace_id}


@app.get("/stream/{trace_id}")
async def stream_events(trace_id: str) -> StreamingResponse:
    """
    SSE stream for a running pipeline. Each event is:
      data: {"event_type": "...", ...payload}\n\n

    Sends `: ping` keep-alives every 25 s so the browser doesn't drop the connection
    mid-pipeline. Closes after the None sentinel (pipeline_done or pipeline_error).

    If called after pipeline_done (within the 5-min TTL) the queue still holds all
    buffered events and the stream drains immediately. After TTL, returns 404.
    """
    q = _event_manager.get_queue(trace_id)
    if q is None:
        raise HTTPException(status_code=404, detail="trace not found or expired")

    async def _generate():
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=25.0)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/result/{trace_id}")
async def get_result(trace_id: str):
    """
    Return the full ContentPack for a completed run.
    202 if still in flight; 404 if not found or expired.
    Used by the UI for run-history load and as a fallback if SSE was missed.
    """
    if trace_id in _running_traces:
        return JSONResponse(status_code=202, content={"status": "running"})
    pack = _result_store.get(trace_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return pack.model_dump()


@app.get("/runs")
async def get_runs():
    """
    Last 20 run summaries from obs.db, projected to the UI's expected shape:
      { trace_id, status, cost_usd, turns_used, brief_goal, timestamp, is_replay, replayed_from }
    RunRecord stores total_cost_usd / start_ts / brief_json — mapped here.
    is_replay and replayed_from are included so the UI can show the replay badge.
    """
    rows = query_runs(app.state.obs_conn, limit=20)
    result = []
    for r in rows:
        try:
            brief = json.loads(r["brief_json"])
            goal = brief.get("goal", "")
            audience = brief.get("audience", "")
            topic = brief.get("topic_focus", "") or ""
            platforms = brief.get("platforms", [])
            product_focus = brief.get("product_focus", [])
        except (json.JSONDecodeError, TypeError):
            goal = audience = topic = ""
            platforms = product_focus = []
        result.append({
            "trace_id": r["trace_id"],
            "status": r["status"],
            "cost_usd": r["total_cost_usd"],
            "turns_used": r["turns_used"],
            "brief_goal": goal,
            "brief_audience": audience,
            "brief_topic": topic,
            "brief_platforms": platforms,
            "brief_products": product_focus,
            "timestamp": r["start_ts"],
            "is_replay": bool(r.get("is_replay", 0)),
            "replayed_from": r.get("replayed_from"),
        })
    return result


@app.get("/runs/{trace_id}/detail")
async def get_run_detail(trace_id: str):
    """
    Full per-agent trace for a completed run.
    Returns: { run, agent_calls, tool_events } where tool_events include input_summary.
    Used by the S7 agent trace section rendered below the result panel.
    """
    detail = query_run_detail(app.state.obs_conn, trace_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"run '{trace_id}' not found")
    return detail


@app.delete("/runs/{trace_id}")
async def delete_run_endpoint(trace_id: str):
    """
    Delete a run and all its telemetry (agent_calls + tool_call_events) from obs.db.
    Also removes the run from the in-memory result store if present.
    Returns 409 if the run is currently in progress.
    """
    if trace_id in _running_traces:
        raise HTTPException(status_code=409, detail="Cannot delete a run that is in progress.")
    removed = delete_run(app.state.obs_conn, trace_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"run '{trace_id}' not found")
    _result_store.pop(trace_id, None)
    return {"deleted": trace_id}


@app.get("/runs/{trace_id}/replay")
async def replay_run(request: Request, trace_id: str):
    """
    Re-execute a past run's brief under a new trace_id.
    Loads brief_json from obs.db, deserializes to ContentBrief, spawns a new pipeline
    background task. Returns immediately with { trace_id, replayed_from }.
    The client should open /stream/{trace_id} for the new run's events.

    Replay contract: same inputs, same corpus state, same code path, fresh trace.
    Not bit-identical — LLM outputs are non-deterministic. Deterministic Validator
    gates (editorial pre-gate, citation-resolution, do_not_discuss) reproduce exactly.
    """
    _check_rate_limit(request.client.host if request.client else "unknown")
    detail = query_run_detail(app.state.obs_conn, trace_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"run '{trace_id}' not found")
    try:
        brief = ContentBrief.model_validate_json(detail["run"]["brief_json"])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not deserialize brief: {exc}")
    new_trace_id = str(uuid.uuid4())
    _event_manager.create(new_trace_id)
    asyncio.create_task(
        _run_pipeline(
            new_trace_id,
            brief,
            app.state,
            is_replay=True,
            replayed_from=trace_id,
        )
    )
    return {"trace_id": new_trace_id, "replayed_from": trace_id}


@app.get("/corpus/chunk/{chunk_id}")
async def get_corpus_chunk(chunk_id: str):
    """
    Return a corpus chunk by ID for the citation click-through viewer.
    The UI highlights cited_substring within content — substring is passed as
    a query param by the UI and stitched back client-side; this endpoint returns
    the raw chunk only.
    """
    chunk = get_chunk(app.state.corpus_conn, chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"chunk '{chunk_id}' not found")
    return chunk.model_dump()
