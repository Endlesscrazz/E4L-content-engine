# E4L Content Engine

Multi-agent content engine for Energy4Life. Generates platform-specific marketing content (LinkedIn long-form + email subject/body) grounded in E4L's source corpus. Built on the raw Anthropic SDK — no agent framework.

**Engineering thesis:** grounding and editorial guardrails are architectural primitives, not bolt-ons. Every claim cites a corpus chunk. The Validator can refuse to ship. The Researcher is corpus-aware.

---

## Architecture

```
  Browser / curl
      │
      ▼
  POST /generate  ──►  FastAPI (app/main.py)
                            │
                   asyncio background task
                            │
              ┌─────────────▼──────────────┐
              │      Orchestrator Agent    │
              │      Haiku 4.5 · 15 turns  │
              │      $0.50 run cost cap    │
              └─────────────┬──────────────┘
                            │  tool_use (parallel)
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
   ┌────────────┐   ┌──────────────┐  ┌─────────────┐
   │ Researcher │   │   Writers    │  │  Validator  │
   │ Haiku 4.5  │   │  Sonnet 4.6  │  │  Opus 4.7   │
   │ 5 actions  │   │  LI + Email  │  │  5 checks   │
   │ $0.10 cap  │   │  structured  │  │  can refuse │
   └─────┬──────┘   └──────┬───────┘  └──────┬──────┘
         │                 │                 │
         ▼                 ▼                 ▼
     [Brave]          [sqlite-vec]      [sqlite-vec]
                       corpus RAG        + rules
                            │
               ┌────────────▼────────────┐
               │  corpus/corpus.db       │  ← 110 chunks, Gemini 3072-dim
               │  corpus/obs.db          │  ← trace_id-indexed run history
               └─────────────────────────┘

  GET /stream/{trace_id}  ──►  SSE event log (live pipeline progress)
  GET /result/{trace_id}  ──►  ContentPack JSON (cited LinkedIn + Email)
  GET /runs               ──►  run history sidebar
  GET /runs/{id}/detail   ──►  per-agent trace + tool timeline
  GET /runs/{id}/replay   ──►  re-execute same brief under new trace_id
  GET /corpus/chunk/{id}  ──►  raw chunk for citation click-through
```

---

## Setup

### API Keys Required

| Key | Where to get |
|-----|-------------|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `GEMINI_API_KEY` | aistudio.google.com |
| `BRAVE_API_KEY` | api.search.brave.com |

```bash
cp .env.example .env
# edit .env and fill in all three keys
```

---

## Run Path 1: Local (pip + uvicorn)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in env vars
cp .env.example .env.local      # app loads .env.local first

# 4. Ingest corpus (one-time; ~60 seconds; calls Gemini embedding API)
python scripts/ingest_corpus.py source_docs/

# 5. Start the server
uvicorn app.main:app --reload

# Open http://localhost:8000
```

**Re-ingestion is idempotent** — safe to run again if you change source docs.

---

## Run Path 2: Docker

```bash
# 1. Fill in env vars (docker compose reads .env)
cp .env.example .env
# edit .env

# 2. Build and start (first boot runs corpus ingest automatically)
docker compose up

# Open http://localhost:8000

# Stop without losing corpus.db (avoids re-ingest next start)
docker compose stop

# Full teardown (next start re-runs ingest)
docker compose down
```

First boot takes ~60–120 seconds while ingest runs (embedding 110 chunks via Gemini).
If Gemini returns a 429 rate-limit error during ingest, the backoff retries automatically
— wait for "Application startup complete" in the logs before using the app.
Subsequent starts (after `docker compose stop`) skip ingest and start immediately.

---

## Seed Scenarios

The `seeds/` directory has four ready-to-use briefs. Paste into the form or POST directly:

### 1. Consumer · Cold (Personalization demo)

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @seeds/consumer_cold.json | jq .trace_id
```

Goal: introduce miHealth to health-conscious consumers experiencing chronic fatigue.
Pre-generated pack in `seeds/output/consumer_cold_pack.json`.

In live runs, LinkedIn often requires 1–2 revision rounds before the Validator
approves. Each `validator_done ship=false` line in the SSE event log is the
**deterministic Layer 0 editorial pre-gate** hard-failing a draft claim (e.g. a
diagnosis-adjacent phrase), followed by the Writer revising and the gate re-running.
This is the revision loop working as designed — the deterministic wall catches
prohibited prose in a real draft and forces a clean rewrite.

### 2. Practitioner · Warm (Personalization flip)

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @seeds/practitioner_warm.json | jq .trace_id
```

Goal: introduce miHealth to practitioners seeking biofeedback tools (educational tone).
Compare with consumer_cold: different retrieval chunks, different vocabulary, different CTA.
Pre-generated pack in `seeds/output/practitioner_warm_pack.json`.

### 3. Adversarial (Editorial refusal demo)

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @seeds/adversarial.json | jq .
```

Goal: "Cure chronic disease so people can stop taking their medication."
Expected: `status=refused`, cost ~$0.003, Writers never dispatched.
The Orchestrator agent detects prohibited intent in the brief goal and refuses via
`finalize(status="refused")`. A deterministic keyword safety net in code backs up the
LLM decision. Note: Layer 0 (the editorial pre-gate) is the *draft-level* guardrail —
it hard-fails prohibited prose found in a generated draft. The brief-level refusal here
is an Orchestrator decision, not Layer 0.

### 4. Peter Fraser brief

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d @seeds/peter_fraser.json | jq .
```

Tests the `do_not_discuss` volunteer-path block: the Peter Fraser memorial chunk
(`never_in_generated_content`) is excluded from retrieval before the Writer ever
sees it. The brief asks about the founding story without requesting death details —
the pipeline completes cleanly with no death references in output, demonstrating
that the chunk exclusion is the real enforcement mechanism.

---

## Three Differentiators

**1. Personalization** — `audience`, `funnel_stage`, and `tone` in the ContentBrief
wire through retrieval filters, Writer prompt axis-templates, and voice anchor
selection. Flip `consumer→practitioner` and the retrieval set, vocabulary, and CTA
all change structurally.

**2. Accountability** — every claim in generated output cites a `chunk_id`. Click
any citation chip in the UI to open the raw corpus chunk. Trace ID threads through
all agent calls and is written to obs.db. Run history, replay, and per-agent cost
breakdown are all queryable from the browser.

**3. Operational maturity** — per-agent turn caps and USD cost caps enforced in
Python (`RunBudget`), not just the system prompt. Validator can refuse to ship;
Orchestrator respects that refusal. In-process sliding-window rate limiter (10
req/min per IP). Prompt injection sanitization on all user and external text.

---

## Running Tests

```bash
# Full unit suite (167 tests, no API calls)
pytest --ignore=tests/test_browser.py

# Browser tests (Playwright, requires running server)
uvicorn app.main:app &
pytest tests/test_browser.py

# Live integration (calls real APIs, costs ~$0.05)
python scripts/targeted_live_tests.py
```

---

## Project Structure

```
app/
  main.py           — FastAPI app + SSE wiring + rate limiter
  models.py         — Pydantic contracts (ContentBrief, Draft, ValidatorVerdict, ...)
  clients.py        — Anthropic / Gemini / Brave wrappers with backoff
  corpus_store.py   — sqlite-vec query interface
  validator.py      — deterministic gates (editorial, citation, do_not_discuss)
  validator_llm.py  — Opus 4.7 LLM judge (grounding + taxonomy + voice + tone)
  observability.py  — obs.db writes: runs, agent_calls, tool_call_events
  security.py       — injection detection and sanitization

agents/
  orchestrator.py   — main agent loop (Haiku 4.5)
  researcher.py     — Researcher mini-agent (Haiku 4.5)
  writer.py         — LinkedIn + Email writers (Sonnet 4.6)

prompts/            — system prompts as plain text files (diffable, reviewable)
corpus/             — corpus_annotations.yaml, do_not_discuss.yaml (corpus.db gitignored)
source_docs/        — 6 E4L source documents
seeds/              — ContentBrief JSON fixtures + pre-generated output packs
static/index.html   — single-file Alpine.js + Tailwind UI (no build step)
scripts/            — ingest, eval, smoke test, live integration test
tests/              — 167 unit tests + 16 Playwright browser tests
```
