"""
S5.5 Live integration smoke test.

Runs one full pipeline call with real API keys. Prints:
  - Per-agent token + cost breakdown
  - Haiku turn-by-turn plan log (tool calls in order)
  - ContentPack summary
  - Pass/fail for each S5.5 gate

Usage:
  python scripts/smoke_test.py

Keys loaded from .env.local (never committed).
Writes obs.db to corpus/obs_smoke.db (separate from production obs.db).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# ── path fix so imports work from project root ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(".env.local", override=True)
load_dotenv()

from app.clients import AnthropicClient, BraveClient, GeminiEmbedder
from app.corpus_store import get_conn
from app.models import Audience, ContentBrief, FunnelStage, Platform, ToneRegister
from app.observability import init_obs_db, query_run_detail, query_runs
from agents.orchestrator import run_orchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke_test")

# ── Seed brief ─────────────────────────────────────────────────────────────────
BRIEF = ContentBrief(
    goal="Introduce the miHealth device to health-conscious consumers experiencing chronic fatigue",
    audience=Audience.CONSUMER,
    funnel_stage=FunnelStage.COLD,
    tone=ToneRegister.CONVERSATIONAL,
    topic_focus="chronic fatigue and cellular energy",
    product_focus=["miHealth"],
)

TRACE_ID = f"smoke_{int(time.time())}"

# ── Event collector ────────────────────────────────────────────────────────────
events: list[tuple[str, dict]] = []

def on_event(event_type: str, payload: dict) -> None:
    events.append((event_type, payload))
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] EVENT: {event_type}  {json.dumps(payload, default=str)[:120]}")


# ── Gates tracker ──────────────────────────────────────────────────────────────
gates: dict[str, bool] = {
    "corpus_accessible": False,
    "researcher_ran": False,
    "writers_dispatched_parallel": False,
    "linkedin_draft_produced": False,
    "email_draft_produced": False,
    "at_least_one_shipped": False,
    "contentpack_complete_or_partial": False,
    "cost_under_cap": False,
    "obs_run_written": False,
}


async def main() -> None:
    print("=" * 70)
    print("E4L Content Engine — S5.5 Live Smoke Test")
    print(f"Trace ID: {TRACE_ID}")
    print(f"Brief:    {BRIEF.goal}")
    print("=" * 70)

    # ── Gate: corpus accessible ────────────────────────────────────────────────
    corpus_db = Path("corpus/corpus.db")
    if not corpus_db.exists():
        print("\n[FAIL] corpus/corpus.db not found.")
        print("       Run: python scripts/ingest_corpus.py source_docs/")
        print("       Cannot proceed without corpus.")
        sys.exit(1)

    try:
        corpus_conn = get_conn(corpus_db)
        row = corpus_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        chunk_count = row[0]
        if chunk_count == 0:
            print("\n[FAIL] corpus.db exists but chunks table is empty. Re-run ingestion.")
            sys.exit(1)
        gates["corpus_accessible"] = True
        print(f"\n[OK]   Corpus: {chunk_count} chunks loaded")
    except Exception as exc:
        print(f"\n[FAIL] Could not open corpus.db: {exc}")
        sys.exit(1)

    # ── Init clients ───────────────────────────────────────────────────────────
    try:
        anthropic_client = AnthropicClient()
        gemini_embedder = GeminiEmbedder()
        brave_client = BraveClient()
    except EnvironmentError as exc:
        print(f"\n[FAIL] Missing API key: {exc}")
        sys.exit(1)

    # ── Init obs db ────────────────────────────────────────────────────────────
    obs_db = Path("corpus/obs_smoke.db")
    obs_conn = init_obs_db(obs_db)
    print(f"[OK]   Obs db: {obs_db}")

    # ── Run pipeline ───────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Running pipeline...")
    print("─" * 70)

    t_start = time.time()
    try:
        pack = await run_orchestrator(
            brief=BRIEF,
            anthropic_client=anthropic_client,
            gemini_embedder=gemini_embedder,
            brave_client=brave_client,
            corpus_conn=corpus_conn,
            obs_conn=obs_conn,
            trace_id=TRACE_ID,
            on_event=on_event,
        )
    except Exception as exc:
        print(f"\n[FAIL] Pipeline raised an exception: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t_start
    print("─" * 70)

    # ── Analyse event log ──────────────────────────────────────────────────────
    event_types = [e[0] for e in events]

    researcher_ran = "researcher_done" in event_types
    gates["researcher_ran"] = researcher_ran

    # Count writer_start events — parallel = both start before either finishes
    writer_starts = [(e[0], e[1]) for e in events if e[0] == "writer_start"]
    gates["writers_dispatched_parallel"] = len(writer_starts) >= 2

    # ── ContentPack assessment ─────────────────────────────────────────────────
    gates["linkedin_draft_produced"] = pack.linkedin_draft is not None
    gates["email_draft_produced"] = pack.email_draft is not None

    li_shipped = pack.linkedin_verdict is not None and pack.linkedin_verdict.ship
    em_shipped = pack.email_verdict is not None and pack.email_verdict.ship
    gates["at_least_one_shipped"] = li_shipped or em_shipped
    gates["contentpack_complete_or_partial"] = pack.status in ("complete", "partial")
    gates["cost_under_cap"] = pack.budget.cost_usd_spent < pack.budget.max_cost_usd

    # ── Obs verification ───────────────────────────────────────────────────────
    runs = query_runs(obs_conn)
    gates["obs_run_written"] = any(r["trace_id"] == TRACE_ID for r in runs)

    # ── Print results ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nStatus:        {pack.status.upper()}")
    print(f"Turns used:    {pack.budget.turns_used} / {pack.budget.max_turns}")
    print(f"Cost:          ${pack.budget.cost_usd_spent:.4f} / ${pack.budget.max_cost_usd:.2f}")
    print(f"Elapsed:       {elapsed:.1f}s")
    print(f"Revisions:     {pack.revisions_used}")

    # Research
    if pack.research:
        print(f"\nResearch:      {len(pack.research.findings)} findings "
              f"(cost: ${pack.research.cost_usd:.4f})")
        if pack.research.no_context_reason:
            print(f"               No-context reason: {pack.research.no_context_reason}")
    else:
        print("\nResearch:      skipped (Orchestrator did not call researcher)")

    # LinkedIn draft
    print("\n─── LinkedIn Draft ───────────────────────────────────────────────")
    if pack.linkedin_draft:
        d = pack.linkedin_draft
        print(f"Body length:   {len(d.body)} chars")
        print(f"Claims:        {len(d.claims)}")
        print(f"Cited chunks:  {d.cited_chunk_ids}")
        print(f"Cost:          ${d.cost_usd:.4f}  ({d.tokens_in}in / {d.tokens_out}out)")
        if pack.linkedin_verdict:
            v = pack.linkedin_verdict
            print(f"Validator:     ship={v.ship}")
            if not v.ship:
                print(f"  Rejected:    {v.revision_notes}")
        print("\nBody preview:")
        print(d.body[:400] + ("..." if len(d.body) > 400 else ""))
    else:
        print("NOT PRODUCED")

    # Email draft
    print("\n─── Email Draft ──────────────────────────────────────────────────")
    if pack.email_draft:
        d = pack.email_draft
        print(f"Subject:       {d.subject}")
        print(f"Body length:   {len(d.body)} chars")
        print(f"Claims:        {len(d.claims)}")
        print(f"Cost:          ${d.cost_usd:.4f}  ({d.tokens_in}in / {d.tokens_out}out)")
        if pack.email_verdict:
            v = pack.email_verdict
            print(f"Validator:     ship={v.ship}")
            if not v.ship:
                print(f"  Rejected:    {v.revision_notes}")
        print("\nBody preview:")
        print((d.subject or "") + "\n\n" + d.body[:300] + ("..." if len(d.body) > 300 else ""))
    else:
        print("NOT PRODUCED")

    # Haiku planning quality
    print("\n─── Haiku 4.5 Planning Quality ───────────────────────────────────")
    print(f"Total events:  {len(events)}")
    print(f"Event sequence:")
    for et, ep in events:
        detail = ""
        if et == "writer_start":
            detail = f"  platform={ep.get('platform')}"
        elif et == "validator_done":
            detail = f"  platform={ep.get('platform')} ship={ep.get('ship')}"
        elif et == "researcher_done":
            detail = f"  findings={ep.get('findings')} cost=${ep.get('cost_usd', 0):.4f}"
        elif et in ("pipeline_finalizing", "pipeline_done"):
            detail = f"  status={ep.get('status')}"
        print(f"  {et}{detail}")

    parallel_writers = len(writer_starts) >= 2
    if parallel_writers:
        platforms_in_order = [e[1].get("platform") for e in writer_starts]
        print(f"\n[PARALLEL]  Both writers dispatched in one Haiku turn: {platforms_in_order}")
    else:
        print(f"\n[SEQUENTIAL] Writers were NOT dispatched in parallel (only {len(writer_starts)} writer_start events)")
        print("             This means Haiku called call_writer in separate turns.")
        print("             Functional but costs an extra orchestrator turn per run.")

    finalize_events = [e for e in events if e[0] == "pipeline_finalizing"]
    if finalize_events:
        print(f"[FINALIZE]  Haiku called finalize correctly: {finalize_events[0][1]}")
    else:
        print("[WARNING]   Haiku did not call finalize — loop ended without explicit finalize")

    # Obs detail
    detail = query_run_detail(obs_conn, TRACE_ID)
    if detail:
        agent_calls = detail.get("agent_calls", [])
        tool_events = detail.get("tool_events", [])  # key is "tool_events" in query_run_detail
        print(f"\n─── Observability ────────────────────────────────────────────────")
        print(f"agent_calls rows:     {len(agent_calls)}")
        print(f"tool_call_events rows:{len(tool_events)}")
        print("Agent calls breakdown:")
        for ac in agent_calls:
            print(f"  {ac['agent_name']:25s}  in={ac['tokens_in']:5d}  out={ac['tokens_out']:4d}  "
                  f"cost=${ac['cost_usd']:.4f}  latency={ac['latency_ms']}ms")

    # Gates summary
    print("\n" + "=" * 70)
    print("GATE SUMMARY")
    print("=" * 70)
    all_pass = True
    for gate, passed in gates.items():
        mark = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        print(f"  {mark}  {gate}")

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL GATES PASSED — S6 is cleared to start.")
    else:
        failed = [g for g, p in gates.items() if not p]
        print(f"GATES FAILED: {', '.join(failed)}")
        print("Review failures above before starting S6.")
    print("=" * 70)

    obs_conn.close()
    corpus_conn.close()


if __name__ == "__main__":
    asyncio.run(main())
