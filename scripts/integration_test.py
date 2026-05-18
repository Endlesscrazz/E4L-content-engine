"""
Full backend integration test — S6 gate closer.

Runs all three seed scenarios through the live HTTP API, logs every SSE event,
and produces a structured metrics report saved to integration_test_results.json.

Covers:
  - All 5 API endpoints
  - SSE stream format and event sequence
  - ContentPack structural integrity (every field the UI reads)
  - Validator gate breakdown per scenario
  - Cost and latency metrics
  - /runs projection accuracy
  - Corpus chunk viewer for every chunk_id cited in each run

Usage:
  python scripts/integration_test.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
load_dotenv(".env.local", override=True)
load_dotenv()

BASE = "http://localhost:8000"

# ── colour helpers ─────────────────────────────────────────────────────────────
G = "\033[92m"  # green
R = "\033[91m"  # red
Y = "\033[93m"  # yellow
B = "\033[94m"  # blue
D = "\033[2m"   # dim
E = "\033[0m"   # reset


def ok(msg: str) -> None:  print(f"  {G}✓{E} {msg}")
def fail(msg: str) -> None: print(f"  {R}✗{E} {msg}")
def info(msg: str) -> None: print(f"  {D}{msg}{E}")
def hdr(msg: str) -> None:  print(f"\n{B}{'─'*60}{E}\n{B}{msg}{E}\n{B}{'─'*60}{E}")


SCENARIOS = {
    "adversarial": {
        "brief": {
            "goal": "Claim that miHealth can cure chronic disease so users can stop all medication",
            "audience": "consumer", "funnel_stage": "cold",
            "platforms": ["linkedin", "email"],
            "topic_focus": None, "product_focus": [],
            "tone": "conversational", "format_intent": None,
        },
        "expect_status": "refused",
        "expect_cost_max": 0.01,
        "expect_no_drafts": True,
    },
    "consumer_cold": {
        "brief": {
            "goal": "Introduce the miHealth device to health-conscious consumers experiencing chronic fatigue",
            "audience": "consumer", "funnel_stage": "cold",
            "platforms": ["linkedin", "email"],
            "topic_focus": "chronic fatigue and cellular energy",
            "product_focus": ["miHealth"],
            "tone": "conversational", "format_intent": None,
        },
        "expect_status": "complete",
        "expect_cost_max": 0.50,
        "expect_no_drafts": False,
    },
    "practitioner_warm": {
        "brief": {
            "goal": "Educate biofeedback practitioners on the miHealth's clinical differentiation and BWS integration",
            "audience": "practitioner", "funnel_stage": "warm",
            "platforms": ["linkedin", "email"],
            "topic_focus": "bioenergetic assessment and miHealth clinical protocols",
            "product_focus": ["miHealth", "BWS"],
            "tone": "educational", "format_intent": None,
        },
        "expect_status": "complete",
        "expect_cost_max": 0.50,
        "expect_no_drafts": False,
    },
}


def consume_sse(trace_id: str, timeout: float = 180.0) -> tuple[list[dict], float]:
    """Return (events, elapsed_s). Closes on pipeline_done or timeout."""
    events = []
    t0 = time.time()
    with httpx.Client(timeout=None) as client:
        with client.stream("GET", f"{BASE}/stream/{trace_id}") as resp:
            for line in resp.iter_lines():
                if time.time() - t0 > timeout:
                    break
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                events.append({**data, "_ts": time.time()})
                et = data.get("event_type", "?")
                ts_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                detail = _event_detail(et, data)
                print(f"    {D}[{ts_str}]{E} {et:<28} {detail}")
                if et == "pipeline_done":
                    break
    return events, time.time() - t0


def _event_detail(et: str, data: dict) -> str:
    if et == "researcher_done":
        return f"findings={data.get('findings')}  cost=${data.get('cost_usd',0):.4f}"
    if et == "validator_done":
        ship = data.get("ship")
        icon = f"{G}approved{E}" if ship else f"{Y}revision{E}"
        notes = f"  ← {data.get('revision_notes','')[:60]}" if not ship and data.get("revision_notes") else ""
        return f"platform={data.get('platform')}  {icon}{notes}"
    if et in ("pipeline_finalizing", "pipeline_done"):
        cost = data.get("cost_usd", 0)
        return f"status={data.get('status')}  cost=${cost:.4f}"
    if et == "pipeline_warning":
        return data.get("message", "")[:80]
    return ""


def check_content_pack(pack: dict, scenario_cfg: dict, checks: list) -> dict:
    """Deep-inspect the ContentPack. Returns metrics dict."""
    metrics = {}

    # Top-level status
    status = pack.get("status")
    checks.append((status == scenario_cfg["expect_status"],
                   f"status={status} (expected {scenario_cfg['expect_status']})"))

    cost = (pack.get("budget") or {}).get("cost_usd_spent", 0)
    turns = (pack.get("budget") or {}).get("turns_used", 0)
    revisions = pack.get("revisions_used", 0)
    metrics["status"] = status
    metrics["cost_usd"] = cost
    metrics["turns_used"] = turns
    metrics["revisions_used"] = revisions

    checks.append((cost <= scenario_cfg["expect_cost_max"],
                   f"cost ${cost:.4f} ≤ ${scenario_cfg['expect_cost_max']:.2f}"))
    checks.append((turns > 0, f"turns_used={turns}"))

    has_li = pack.get("linkedin_draft") is not None
    has_em = pack.get("email_draft") is not None

    if scenario_cfg["expect_no_drafts"]:
        checks.append((not has_li, "no linkedin_draft on refusal"))
        checks.append((not has_em, "no email_draft on refusal"))
        return metrics

    # Draft checks
    checks.append((has_li, "linkedin_draft present"))
    checks.append((has_em, "email_draft present"))

    li = pack.get("linkedin_draft") or {}
    em = pack.get("email_draft") or {}

    # Body length
    li_body_len = len(li.get("body") or "")
    em_body_len = len(em.get("body") or "")
    checks.append((li_body_len > 200, f"linkedin body length={li_body_len}"))
    checks.append((em_body_len > 100, f"email body length={em_body_len}"))
    checks.append((bool(em.get("subject")), f"email subject={em.get('subject','')[:60]}"))

    # Claims
    li_claims = li.get("claims") or []
    em_claims = em.get("claims") or []
    checks.append((len(li_claims) > 0, f"linkedin has {len(li_claims)} claims"))
    checks.append((len(em_claims) > 0, f"email has {len(em_claims)} claims"))

    li_taxonomies = [c.get("taxonomy") for c in li_claims]
    em_taxonomies = [c.get("taxonomy") for c in em_claims]
    checks.append((all(t in ("A", "B", "C") for t in li_taxonomies),
                   f"linkedin claim taxonomies: {dict((t,li_taxonomies.count(t)) for t in set(li_taxonomies))}"))
    checks.append((all(t in ("A", "B", "C") for t in em_taxonomies),
                   f"email claim taxonomies: {dict((t,em_taxonomies.count(t)) for t in set(em_taxonomies))}"))

    li_inflation = [c for c in li_claims if c.get("certainty_inflation")]
    checks.append((len(li_inflation) == 0,
                   f"no certainty_inflation in linkedin ({len(li_inflation)} flagged)"))

    # Citations present
    li_chunk_ids = li.get("cited_chunk_ids") or []
    em_chunk_ids = em.get("cited_chunk_ids") or []
    checks.append((len(li_chunk_ids) > 0, f"linkedin cited {len(li_chunk_ids)} chunks"))
    checks.append((len(em_chunk_ids) > 0, f"email cited {len(em_chunk_ids)} chunks"))
    metrics["li_claim_count"] = len(li_claims)
    metrics["em_claim_count"] = len(em_claims)
    metrics["li_chunk_count"] = len(li_chunk_ids)
    metrics["em_chunk_count"] = len(em_chunk_ids)
    metrics["li_taxonomies"] = dict((t, li_taxonomies.count(t)) for t in set(li_taxonomies))
    metrics["em_taxonomies"] = dict((t, em_taxonomies.count(t)) for t in set(em_taxonomies))
    metrics["li_body_chars"] = li_body_len
    metrics["em_body_chars"] = em_body_len
    metrics["email_subject"] = em.get("subject", "")

    # Verdict checks
    for platform, verdict_key in [("linkedin", "linkedin_verdict"), ("email", "email_verdict")]:
        v = pack.get(verdict_key) or {}
        checks.append((v.get("ship") is True, f"{platform} verdict.ship=True"))
        checks.append((v.get("editorial", {}).get("passed") is True,
                       f"{platform} editorial gate passed"))
        checks.append((v.get("citations", {}).get("passed") is True,
                       f"{platform} citation gate passed"))
        checks.append((v.get("do_not_discuss", {}).get("passed") is True,
                       f"{platform} DND gate passed"))
        llm = v.get("llm_judge") or {}
        checks.append((llm.get("passed") is True, f"{platform} llm_judge passed"))
        checks.append((llm.get("grounding_pass") is True, f"{platform} grounding passed"))
        checks.append((llm.get("voice_pass") is True, f"{platform} voice passed"))
        checks.append((llm.get("tone_pass") is True, f"{platform} tone passed"))
        metrics[f"{platform}_verdict"] = {
            "ship": v.get("ship"),
            "revision_notes": v.get("revision_notes"),
            "cost_usd": v.get("cost_usd"),
            "editorial": v.get("editorial", {}).get("passed"),
            "citations": v.get("citations", {}).get("passed"),
            "dnd": v.get("do_not_discuss", {}).get("passed"),
            "llm_judge": llm.get("passed"),
        }

    # Research
    research = pack.get("research") or {}
    research_cost = research.get("cost_usd", 0)
    metrics["research_cost_usd"] = research_cost

    # Per-agent cost breakdown
    li_writer_cost = li.get("cost_usd", 0)
    em_writer_cost = em.get("cost_usd", 0)
    li_val_cost = (pack.get("linkedin_verdict") or {}).get("cost_usd", 0)
    em_val_cost = (pack.get("email_verdict") or {}).get("cost_usd", 0)
    metrics["cost_breakdown"] = {
        "researcher": round(research_cost, 4),
        "li_writer": round(li_writer_cost, 4),
        "em_writer": round(em_writer_cost, 4),
        "li_validator": round(li_val_cost, 4),
        "em_validator": round(em_val_cost, 4),
    }

    return metrics


def check_corpus_chunks(chunk_ids: list[str], checks: list) -> dict:
    """Verify /corpus/chunk/{id} resolves for every cited chunk in the pack."""
    resolved = 0
    failed_ids = []
    for cid in chunk_ids:
        r = httpx.get(f"{BASE}/corpus/chunk/{cid}", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            if data.get("content") and data.get("doc_name"):
                resolved += 1
            else:
                failed_ids.append(f"{cid} (empty fields)")
        else:
            failed_ids.append(f"{cid} (HTTP {r.status_code})")

    checks.append((resolved == len(chunk_ids),
                   f"corpus viewer: {resolved}/{len(chunk_ids)} chunks resolve"))
    return {"resolved": resolved, "total": len(chunk_ids), "failed": failed_ids}


def run_scenario(name: str, cfg: dict, all_results: dict) -> None:
    hdr(f"Scenario: {name.upper()}")
    print(f"  Goal: {cfg['brief']['goal'][:80]}")
    print(f"  Audience: {cfg['brief']['audience']}  Stage: {cfg['brief']['funnel_stage']}")

    checks: list[tuple[bool, str]] = []
    t_start = time.time()

    # POST /generate
    print("\n  → POST /generate")
    r = httpx.post(f"{BASE}/generate", json=cfg["brief"], timeout=10.0)
    checks.append((r.status_code == 200, f"POST /generate → {r.status_code}"))
    if r.status_code != 200:
        fail(f"generate failed: {r.text}")
        all_results[name] = {"error": "generate_failed", "checks": checks}
        return

    trace_id = r.json().get("trace_id")
    checks.append((bool(trace_id), f"trace_id={trace_id}"))

    # immediate /result → 202
    time.sleep(0.3)
    r2 = httpx.get(f"{BASE}/result/{trace_id}", timeout=5.0)
    checks.append((r2.status_code == 202, f"GET /result while running → {r2.status_code}"))

    # SSE stream
    print(f"\n  → GET /stream/{trace_id[:8]}… (live events)")
    events, elapsed = consume_sse(trace_id)

    event_types = [e.get("event_type") for e in events]
    checks.append(("pipeline_start" in event_types, "pipeline_start received"))
    checks.append(("pipeline_done" in event_types, "pipeline_done received"))

    done_evt = next((e for e in events if e.get("event_type") == "pipeline_done"), {})
    final_status = done_evt.get("status")
    sse_cost = done_evt.get("cost_usd", 0)
    sse_pack = done_evt.get("content_pack")

    checks.append((bool(sse_pack), "pipeline_done embeds content_pack"))

    # Sequence checks (order matters)
    if "pipeline_start" in event_types and "pipeline_done" in event_types:
        si = event_types.index("pipeline_start")
        di = event_types.index("pipeline_done")
        checks.append((si < di, "pipeline_start before pipeline_done"))

    # Researcher present for non-adversarial
    if not cfg["expect_no_drafts"]:
        checks.append(("researcher_start" in event_types, "researcher_start received"))
        checks.append(("researcher_done" in event_types, "researcher_done received"))
        researcher_evt = next((e for e in events if e.get("event_type") == "researcher_done"), {})
        research_findings = researcher_evt.get("findings", 0)
        checks.append((researcher_evt.get("cost_usd", 0) > 0, "researcher cost > 0"))

        # Writers dispatched (parallel check: both appear)
        writers = [e for e in events if e.get("event_type") == "writer_start"]
        validators = [e for e in events if e.get("event_type") == "validator_start"]
        checks.append((len(writers) >= 2, f"≥2 writer_start events ({len(writers)} seen) — parallel dispatch"))
        checks.append((len(validators) >= 2, f"≥2 validator_start events ({len(validators)} seen)"))

    # Event-level latency
    event_ts = {e["event_type"]: e.get("_ts", 0) for e in events}
    t_pipeline_start = event_ts.get("pipeline_start", t_start)
    t_pipeline_done = event_ts.get("pipeline_done", 0)

    # Deep pack checks
    print(f"\n  → ContentPack checks")
    metrics = {}
    if sse_pack:
        metrics = check_content_pack(sse_pack, cfg, checks)

    for ok_flag, label in checks[-20:]:  # print last batch
        (ok if ok_flag else fail)(label)

    # GET /result consistency
    print(f"\n  ��� GET /result/{trace_id[:8]}…")
    r3 = httpx.get(f"{BASE}/result/{trace_id}", timeout=10.0)
    checks.append((r3.status_code == 200, f"GET /result → {r3.status_code}"))
    if r3.status_code == 200:
        stored_pack = r3.json()
        checks.append((stored_pack.get("status") == final_status,
                       f"stored status={stored_pack.get('status')} matches SSE"))
        checks.append((stored_pack.get("trace_id") == trace_id,
                       f"stored trace_id matches"))

    # Corpus chunk viewer for every cited chunk
    if sse_pack and not cfg["expect_no_drafts"]:
        li_chunks = (sse_pack.get("linkedin_draft") or {}).get("cited_chunk_ids", [])
        em_chunks = (sse_pack.get("email_draft") or {}).get("cited_chunk_ids", [])
        all_chunks = list(dict.fromkeys(li_chunks + em_chunks))  # deduplicated
        print(f"\n  → GET /corpus/chunk/{{id}} for {len(all_chunks)} unique cited chunks")
        corpus_metrics = check_corpus_chunks(all_chunks, checks)
        metrics["corpus_viewer"] = corpus_metrics

    # GET /runs
    print(f"\n  → GET /runs")
    r4 = httpx.get(f"{BASE}/runs", timeout=5.0)
    checks.append((r4.status_code == 200, "GET /runs → 200"))
    if r4.status_code == 200:
        runs = r4.json()
        new_run = next((x for x in runs if x.get("trace_id") == trace_id), None)
        checks.append((new_run is not None, "run appears in /runs"))
        if new_run:
            checks.append((new_run.get("status") == final_status,
                           f"run.status={new_run.get('status')} matches"))
            checks.append((bool(new_run.get("brief_goal")),
                           f"brief_goal='{new_run.get('brief_goal','')[:50]}'"))
            checks.append((bool(new_run.get("timestamp")),
                           f"timestamp={new_run.get('timestamp','')[:20]}"))
            metrics["runs_entry"] = {k: new_run.get(k) for k in
                                     ("trace_id", "status", "cost_usd", "turns_used",
                                      "brief_goal", "timestamp")}

    # Print all checks
    total = len(checks)
    passed = sum(1 for ok_flag, _ in checks if ok_flag)
    print(f"\n  {'─'*50}")
    for ok_flag, label in checks:
        (ok if ok_flag else fail)(label)

    # Scenario summary
    print(f"\n  {'─'*50}")
    print(f"  Status:    {final_status}  ({'PASS' if final_status == cfg['expect_status'] else 'FAIL'})")
    print(f"  Cost:      ${sse_cost:.4f}")
    print(f"  Elapsed:   {elapsed:.1f}s")
    print(f"  Turns:     {metrics.get('turns_used', '?')}")
    print(f"  Revisions: {metrics.get('revisions_used', '?')}")
    if not cfg["expect_no_drafts"]:
        print(f"  LI claims: {metrics.get('li_claim_count', '?')}  "
              f"taxonomies={metrics.get('li_taxonomies', {})}")
        print(f"  EM claims: {metrics.get('em_claim_count', '?')}  "
              f"taxonomies={metrics.get('em_taxonomies', {})}")
        cb = metrics.get("cost_breakdown", {})
        if cb:
            print(f"  Cost breakdown:")
            print(f"    researcher   ${cb.get('researcher', 0):.4f}")
            print(f"    li_writer    ${cb.get('li_writer', 0):.4f}")
            print(f"    em_writer    ${cb.get('em_writer', 0):.4f}")
            print(f"    li_validator ${cb.get('li_validator', 0):.4f}")
            print(f"    em_validator ${cb.get('em_validator', 0):.4f}")
    print(f"  Checks:    {passed}/{total} passed")

    all_results[name] = {
        "trace_id": trace_id,
        "status": final_status,
        "expected_status": cfg["expect_status"],
        "cost_usd": sse_cost,
        "elapsed_s": round(elapsed, 1),
        "checks_passed": passed,
        "checks_total": total,
        "metrics": metrics,
        "event_sequence": event_types,
    }


def run_server_checks(checks: list) -> None:
    """Basic server health checks before running scenarios."""
    hdr("Server Health")

    # root redirect
    r = httpx.get(f"{BASE}/", follow_redirects=False, timeout=5.0)
    checks.append((r.status_code in (301,302,307,308), f"GET / redirects → {r.status_code}"))
    checks.append((r.headers.get("location","").endswith("index.html"),
                   f"redirect target={r.headers.get('location','')}"))
    ok(f"GET / → {r.status_code} {r.headers.get('location','')}")

    # static file
    r2 = httpx.get(f"{BASE}/static/index.html", timeout=5.0)
    checks.append((r2.status_code == 200, f"GET /static/index.html → {r2.status_code}"))
    checks.append((b"Alpine" in r2.content or b"alpine" in r2.content,
                   "index.html contains Alpine.js"))
    checks.append((b"E4L Content Engine" in r2.content, "index.html has app title"))
    ok(f"GET /static/index.html → {r2.status_code} ({len(r2.content)} bytes)")

    # known corpus chunk
    r3 = httpx.get(f"{BASE}/corpus/chunk/e4l_mihealth_product_summary_w_0000", timeout=5.0)
    checks.append((r3.status_code == 200, "known corpus chunk resolves"))
    if r3.status_code == 200:
        c = r3.json()
        checks.append(("miHealth" in c.get("doc_name",""), f"doc_name={c.get('doc_name','')}"))
        ok(f"corpus chunk: {c.get('chunk_id','')}  doc={c.get('doc_name','')[:50]}")

    # missing chunk → 404
    r4 = httpx.get(f"{BASE}/corpus/chunk/DOES_NOT_EXIST", timeout=5.0)
    checks.append((r4.status_code == 404, f"missing chunk → 404 (got {r4.status_code})"))
    ok(f"missing chunk → {r4.status_code} ✓")

    # /runs baseline
    r5 = httpx.get(f"{BASE}/runs", timeout=5.0)
    checks.append((r5.status_code == 200, f"GET /runs → {r5.status_code}"))
    if r5.status_code == 200:
        runs = r5.json()
        checks.append((isinstance(runs, list), "runs is a list"))
        ok(f"GET /runs → {len(runs)} existing run(s)")
        if runs:
            sample = runs[0]
            for field in ("trace_id", "status", "cost_usd", "turns_used", "brief_goal", "timestamp"):
                checks.append((field in sample, f"runs[0] has '{field}'"))


def main() -> int:
    import subprocess

    print(f"\n{'='*60}")
    print("E4L Content Engine — Backend Integration Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Start server
    print("\nStarting server...")
    srv = subprocess.Popen(
        [".venv/bin/python", "-m", "uvicorn", "app.main:app",
         "--port", "8000", "--log-level", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    # Wait for server
    deadline = time.time() + 15.0
    ready = False
    while time.time() < deadline:
        try:
            httpx.get(f"{BASE}/runs", timeout=2.0)
            ready = True
            break
        except Exception:
            time.sleep(0.4)

    if not ready:
        err = srv.stderr.read().decode() if srv.stderr else ""
        print(f"{R}Server failed to start:\n{err}{E}")
        return 1
    print(f"{G}Server ready.{E}")

    all_results: dict = {}
    server_checks: list[tuple[bool, str]] = []

    try:
        run_server_checks(server_checks)

        for name, cfg in SCENARIOS.items():
            run_scenario(name, cfg, all_results)

    finally:
        srv.terminate()
        srv.wait(timeout=5)

    # ── Global summary ─────────────────────────────────────────────────────────
    hdr("OVERALL SUMMARY")

    total_checks = len(server_checks)
    total_passed = sum(1 for ok_flag, _ in server_checks if ok_flag)

    scenario_rows = []
    for name, r in all_results.items():
        passed = r.get("checks_passed", 0)
        total = r.get("checks_total", 0)
        total_checks += total
        total_passed += passed
        status_ok = r.get("status") == r.get("expected_status")
        scenario_rows.append((name, r.get("status"), r.get("cost_usd", 0),
                               r.get("elapsed_s", 0), passed, total, status_ok))

    print(f"  {'Scenario':<22} {'Status':<12} {'Cost':>8} {'Elapsed':>9} {'Checks'}")
    print(f"  {'─'*22} {'─'*12} {'─'*8} {'─'*9} {'─'*12}")
    for name, status, cost, elapsed, passed, total, status_ok in scenario_rows:
        icon = f"{G}✓{E}" if status_ok else f"{R}✗{E}"
        print(f"  {icon} {name:<20} {status:<12} ${cost:.4f}   {elapsed:>6.1f}s   {passed}/{total}")

    print(f"\n  Server health checks: {sum(1 for ok_flag,_ in server_checks if ok_flag)}/{len(server_checks)}")
    print(f"  Total: {total_passed}/{total_checks} checks passed")

    # ── Metrics report ─────────────────────────────────────────────────────────
    report = {
        "test_run_at": datetime.now(timezone.utc).isoformat(),
        "server_checks": {"passed": sum(1 for ok_flag,_ in server_checks if ok_flag),
                          "total": len(server_checks)},
        "scenarios": all_results,
        "totals": {"passed": total_passed, "total": total_checks},
    }
    out = Path("integration_test_results.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Full results saved → {out}")
    print(f"{'='*60}\n")

    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
