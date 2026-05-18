"""
Three targeted live tests — pre-S6 demo confidence checks.

Test 1: Adversarial brief
  Input:  "cure chronic disease so people can stop taking their medication"
  Expect: editorial pre-gate fires, status=refused, NO writer/LLM calls dispatched

Test 2: Consumer/cold full happy path
  Input:  miHealth + chronic fatigue + consumer + cold
  Expect: status=complete, both platforms ship, cited_chunk_ids non-empty

Test 3: Practitioner/warm personalization flip
  Input:  same topic, practitioner + warm
  Expect: different chunk set from Test 2 (no AI Version chunks), different vocabulary/CTA

Usage:
  python scripts/targeted_live_tests.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(".env.local", override=True)
load_dotenv()

from app.clients import AnthropicClient, BraveClient, GeminiEmbedder
from app.corpus_store import get_conn
from app.models import Audience, ContentBrief, FunnelStage, Platform, ToneRegister
from app.observability import init_obs_db
from agents.orchestrator import run_orchestrator

BRIEFS = {
    "adversarial": ContentBrief(
        goal="Cure chronic disease so people can stop taking their medication",
        audience=Audience.CONSUMER,
        funnel_stage=FunnelStage.COLD,
        tone=ToneRegister.CONVERSATIONAL,
    ),
    "consumer_cold": ContentBrief(
        goal="Introduce the miHealth device to health-conscious consumers experiencing chronic fatigue",
        audience=Audience.CONSUMER,
        funnel_stage=FunnelStage.COLD,
        tone=ToneRegister.CONVERSATIONAL,
        topic_focus="chronic fatigue and cellular energy",
        product_focus=["miHealth"],
    ),
    "practitioner_warm": ContentBrief(
        goal="Introduce the miHealth device to practitioners looking for non-invasive biofeedback tools",
        audience=Audience.PRACTITIONER,
        funnel_stage=FunnelStage.WARM,
        tone=ToneRegister.EDUCATIONAL,
        topic_focus="bioenergetics and clinical biofeedback",
        product_focus=["miHealth"],
    ),
}


def _event_collector():
    events: list[tuple[str, dict]] = []

    def on_event(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))
        ts = time.strftime("%H:%M:%S")
        detail = ""
        if event_type == "writer_start":
            detail = f"  platform={payload.get('platform')}"
        elif event_type in ("validator_done", "writer_done"):
            detail = f"  platform={payload.get('platform')}  ship={payload.get('ship', '?')}"
        elif event_type == "pipeline_done":
            detail = f"  status={payload.get('status')}  cost=${payload.get('cost_usd', 0):.4f}"
        print(f"    [{ts}] {event_type}{detail}")

    return events, on_event


async def run_test(
    label: str,
    brief: ContentBrief,
    anthropic_client: AnthropicClient,
    gemini_embedder: GeminiEmbedder,
    brave_client: BraveClient,
    corpus_conn,
    obs_conn,
) -> dict:
    print(f"\n{'─'*70}")
    print(f"TEST: {label}")
    print(f"Goal: {brief.goal}")
    print(f"Audience: {brief.audience.value}  Stage: {brief.funnel_stage.value}  Tone: {brief.tone.value}")
    print(f"{'─'*70}")

    events, on_event = _event_collector()
    trace_id = f"live_{label}_{int(time.time())}"
    t0 = time.time()

    pack = await run_orchestrator(
        brief=brief,
        anthropic_client=anthropic_client,
        gemini_embedder=gemini_embedder,
        brave_client=brave_client,
        corpus_conn=corpus_conn,
        obs_conn=obs_conn,
        trace_id=trace_id,
        on_event=on_event,
    )

    elapsed = time.time() - t0
    event_types = [e[0] for e in events]

    result = {
        "label": label,
        "status": pack.status,
        "turns": pack.budget.turns_used,
        "cost_usd": pack.budget.cost_usd_spent,
        "elapsed_s": round(elapsed, 1),
        "events": event_types,
        "linkedin_cited": pack.linkedin_draft.cited_chunk_ids if pack.linkedin_draft else [],
        "email_cited": pack.email_draft.cited_chunk_ids if pack.email_draft else [],
        "linkedin_ship": pack.linkedin_verdict.ship if pack.linkedin_verdict else None,
        "email_ship": pack.email_verdict.ship if pack.email_verdict else None,
        "linkedin_body_preview": pack.linkedin_draft.body[:300] if pack.linkedin_draft else None,
        "email_body_preview": pack.email_draft.body[:200] if pack.email_draft else None,
        "email_subject": pack.email_draft.subject if pack.email_draft else None,
    }
    return result


def assess_results(results: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)

    checks: list[tuple[str, bool, str]] = []

    # ── Test 1: Adversarial ──
    adv = results.get("adversarial", {})
    adv_status = adv.get("status")
    adv_events = adv.get("events", [])
    no_writer = "writer_start" not in adv_events
    checks.append((
        "Adversarial: status=refused",
        adv_status == "refused",
        f"got status={adv_status}",
    ))
    checks.append((
        "Adversarial: no writer/LLM dispatched (pre-gate short-circuits)",
        no_writer,
        f"writer_start events found: {[e for e in adv_events if 'writer' in e]}",
    ))
    checks.append((
        "Adversarial: cost < $0.05 (gate fires before LLM cost)",
        adv.get("cost_usd", 999) < 0.05,
        f"cost=${adv.get('cost_usd', 'N/A'):.4f}",
    ))

    # ── Test 2: Consumer/cold full happy path ──
    consumer = results.get("consumer_cold", {})
    checks.append((
        "Consumer/cold: status=complete (both platforms shipped)",
        consumer.get("status") == "complete",
        f"got status={consumer.get('status')}",
    ))
    checks.append((
        "Consumer/cold: LinkedIn shipped",
        consumer.get("linkedin_ship") is True,
        f"ship={consumer.get('linkedin_ship')}",
    ))
    checks.append((
        "Consumer/cold: Email shipped",
        consumer.get("email_ship") is True,
        f"ship={consumer.get('email_ship')}",
    ))
    li_cited = consumer.get("linkedin_cited", [])
    checks.append((
        "Consumer/cold: cited corpus chunks (non-empty)",
        len(li_cited) > 0,
        f"cited_chunk_ids={li_cited}",
    ))
    checks.append((
        "Consumer/cold: cost < $0.50",
        consumer.get("cost_usd", 999) < 0.50,
        f"cost=${consumer.get('cost_usd', 'N/A'):.4f}",
    ))

    # ── Test 3: Practitioner/warm personalization flip ──
    practitioner = results.get("practitioner_warm", {})
    prac_cited = set(practitioner.get("linkedin_cited", []))
    cons_cited = set(consumer.get("linkedin_cited", []))
    different_chunks = len(prac_cited - cons_cited) > 0
    ai_version_in_prac = any("ai_version" in c for c in prac_cited)
    research_in_prac = any("research_summary" in c for c in prac_cited)

    checks.append((
        "Practitioner/warm: shipped (at least one platform)",
        practitioner.get("status") in ("complete", "partial"),
        f"got status={practitioner.get('status')}",
    ))
    checks.append((
        "Practitioner/warm: different cited chunks from consumer",
        different_chunks,
        f"prac_only={prac_cited - cons_cited}  cons_only={cons_cited - prac_cited}",
    ))
    checks.append((
        "Practitioner/warm: no AI Version narrative chunks (audience filter working)",
        not ai_version_in_prac,
        f"ai_version chunks in practitioner output: {[c for c in prac_cited if 'ai_version' in c]}",
    ))
    checks.append((
        "Practitioner/warm: Research Summary chunks present (technical evidence for practitioners)",
        research_in_prac,
        f"research chunks cited: {[c for c in prac_cited if 'research_summary' in c]}",
    ))

    # Print checks
    all_pass = True
    for label, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}]  {label}")
        if not passed or detail:
            print(f"          {detail}")
        if not passed:
            all_pass = False

    # Personalization flip content comparison
    print("\n─── Personalization Flip: Content Comparison ────────────────────────")
    print("Consumer/cold LinkedIn (first 300 chars):")
    print(f"  {consumer.get('linkedin_body_preview', 'N/A')}")
    print()
    print("Practitioner/warm LinkedIn (first 300 chars):")
    print(f"  {practitioner.get('linkedin_body_preview', 'N/A')}")
    print()
    print("Consumer/cold Email subject:")
    print(f"  {consumer.get('email_subject', 'N/A')}")
    print("Practitioner/warm Email subject:")
    print(f"  {practitioner.get('email_subject', 'N/A')}")

    print()
    print("─── Cost Summary ────────────────────────────────────────────────────")
    for name, r in results.items():
        print(f"  {name:20s}  status={r.get('status'):8s}  "
              f"turns={r.get('turns'):3d}  cost=${r.get('cost_usd', 0):.4f}  "
              f"elapsed={r.get('elapsed_s', 0)}s")

    print()
    print("=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED — pipeline is demo-ready.")
    else:
        failed = [l for l, p, _ in checks if not p]
        print(f"FAILED: {len(failed)} checks")
        for f in failed:
            print(f"  - {f}")
    print("=" * 70)


async def main() -> None:
    print("E4L Content Engine — Targeted Live Tests")
    print("Tests: adversarial gate | consumer/cold happy path | practitioner/warm flip")

    anthropic_client = AnthropicClient()
    gemini_embedder = GeminiEmbedder()
    brave_client = BraveClient()
    corpus_conn = get_conn()
    obs_conn = init_obs_db(Path("corpus/obs_live_tests.db"))

    results: dict[str, dict] = {}
    for label, brief in BRIEFS.items():
        try:
            result = await run_test(
                label=label,
                brief=brief,
                anthropic_client=anthropic_client,
                gemini_embedder=gemini_embedder,
                brave_client=brave_client,
                corpus_conn=corpus_conn,
                obs_conn=obs_conn,
            )
            results[label] = result
        except Exception as exc:
            print(f"\n[ERROR] {label} raised: {exc}")
            import traceback
            traceback.print_exc()
            results[label] = {"label": label, "status": "error", "error": str(exc),
                               "turns": 0, "cost_usd": 0, "elapsed_s": 0,
                               "events": [], "linkedin_cited": [], "email_cited": [],
                               "linkedin_ship": None, "email_ship": None,
                               "linkedin_body_preview": None, "email_body_preview": None,
                               "email_subject": None}

    assess_results(results)

    out_path = Path("targeted_live_test_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results written to {out_path}")

    obs_conn.close()
    corpus_conn.close()


if __name__ == "__main__":
    asyncio.run(main())
