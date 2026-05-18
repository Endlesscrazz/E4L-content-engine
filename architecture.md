# E4L Content Engine — Architecture
# Generated: 2026-05-15 by Side Projects Agent (revision 2)
# → Claude Code CLI reviews and may suggest modifications before building
# ─────────────────────────────────────────────────────────────────────

## COMPONENT DIAGRAM

```
                      ┌──────────────────────────┐
                      │   Frontend (HTML + SSE)  │
                      │   - personalization knobs│
                      │   - live agent activity  │
                      │   - click-through cites  │
                      │   - run replay browser   │
                      │   - cost telemetry footer│
                      └────────────┬─────────────┘
                                   │  HTTP + SSE
                                   ▼
                      ┌──────────────────────────┐
                      │   FastAPI Backend        │
                      │   /generate /stream      │
                      │   /result /runs          │
                      │   /corpus/chunk          │
                      │   trace-id middleware    │
                      │   rate limit + sanitize  │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                   ┌────────────────────────────────┐
                   │   Orchestrator Agent           │
                   │   (Haiku 4.5, temp 0.3)        │
                   │   - plans content pack         │
                   │   - dispatches specialists     │
                   │   - parallel Writers           │
                   │   - revision loop (max 3)      │
                   │   - turn cap: 15               │
                   │   - cost cap: $0.50/run        │
                   └────────────┬───────────────────┘
                                │
          tool_use (parallel where supported)
   ┌───────────────────────────┼───────────────────────────┐
   ▼                           ▼                           ▼
┌─────────────────┐      ┌──────────────┐         ┌──────────────────┐
│ Researcher      │      │   Writers    │         │   Validator      │
│ mini-agent      │      │  (LinkedIn,  │         │   (Opus 4.7)     │
│ (own loop)      │      │   Email)     │         │   temp 0.0       │
│                 │      │              │         │                  │
│ - broad_search  │      │  - templated │         │  - claim         │
│ - read_url      │      │    per axis  │         │    taxonomy      │
│ - score_corpus_ │      │  - cites     │         │    A/B/C/D       │
│   relevance     │      │    required  │         │  - 5 checks      │
│ - 5 action cap  │      │  - temp 0.7  │         │  - hybrid:       │
│ - $0.10 cap     │      │              │         │    regex + LLM   │
└────────┬────────┘      └──────┬───────┘         └────────┬─────────┘
         │                      │                           │
         ▼                      ▼                           ▼
  [Brave Search]         [Source RAG]            [Source RAG +
                                                  editorial rules +
                                                  voice anchors]
                                  │
                                  ▼
                     ┌──────────────────────────┐
                     │   sqlite-vec corpus      │
                     │   - chunks + metadata    │
                     │   - Gemini 3072-dim      │
                     │   - voice anchor index   │
                     │   - do_not_discuss flags │
                     │   - structural patterns  │
                     └──────────────────────────┘

                     ┌──────────────────────────┐
                     │   Observability sqlite   │
                     │   - runs table           │
                     │   - agent_calls table    │
                     │   - tool_call_events     │
                     │   indexed by trace_id    │
                     └──────────────────────────┘
```

Data flow (typical run):
1. User submits goal + personalization params via frontend → FastAPI → Orchestrator
2. Orchestrator decides plan; calls research_topic
3. Orchestrator calls retrieve_source_chunks to build a ContentBrief
4. Orchestrator dispatches Writers in parallel (LinkedIn + Email)
5. Orchestrator calls Validator on each draft
6. On fail: Orchestrator calls Writer again with revision_notes (max 3 rounds)
7. All pass or turn/cost cap hit: finalize_pack
8. Frontend streams intermediate events throughout via SSE

## TECH STACK

| Component        | Technology                    | Why                                                  | Depth         |
|------------------|-------------------------------|------------------------------------------------------|---------------|
| Orchestration    | Anthropic SDK + tool use      | Raw primitives, OpenClaw-aligned, no framework dep   | MEDIUM        |
| Embeddings       | Gemini 3072-dim               | Same model as Startup Compass                        | HIGH          |
| Vector store     | sqlite-vec                    | Single file, zero infra, same as context-bridge      | HIGH          |
| Structured I/O   | pydantic                      | Typed tool contracts; not a framework                | HIGH          |
| Backend          | FastAPI                       | Async, SSE-friendly                                  | MEDIUM-HIGH   |
| Frontend         | Vanilla HTML/JS + SSE         | No build step; minimal surface area                  | MEDIUM        |
| External search  | Brave Search API              | Visible agentic tool                                 | LOW (new)     |
| Source LLMs      | Haiku 4.5 (Orch/Researcher) + Sonnet 4.6 (Writer) + Opus 4.7 (Validator) | Cost/quality balance per agent tier | MEDIUM-HIGH   |
| Persistence      | sqlite (corpus + obs)         | One file each, no external deps                      | HIGH          |
| Container        | Docker Compose                | Reproducibility signal for AI Infra role             | MEDIUM        |

## THE KEY ARCHITECTURAL DECISION

**Decision:** Build orchestration on raw Anthropic SDK tool use rather than
adopting an agent framework (LangGraph, CrewAI, AutoGen, Swarm). The
Orchestrator is a Claude model whose "tools" are functions that wrap calls
to specialist Claude models. Specialists communicate via typed pydantic
contracts, not free-form messages.

**Alternatives considered:**
- **LangGraph:** rejected. Adds dependency surface; no prior depth to defend
  framework-internal decisions in code review. Energy4Life builds OpenClaw —
  their own framework. Arriving with LangGraph telegraphs wrong intuition
  about abstraction layer.
- **CrewAI:** rejected. "Agents in roles" abstraction obscures protocol layer.
- **Single LLM with all tools attached** (no specialists): rejected. Collapses
  multi-agent framing the JD asks for. Loses typed contracts.
- **Autonomous outer loop** (Option C two-loop): deferred to Phase 2.
  A one-shot demo can't show its value. Researcher is autonomous internally;
  outer continuous loop is not.

**Trade-off accepted:** More code than picking up CrewAI. No prebuilt retry /
state / concurrency primitives. Risk: behavior harder to constrain than a
framework's hardcoded flow.

**Interview defense (verbatim):**
"I built orchestration on raw Anthropic SDK with my own minimal coordination
layer rather than LangGraph or CrewAI. Three reasons. First, you're building
OpenClaw — your own framework — so adding a framework dependency would have
signaled wrong intuition about abstraction layers. Second, at this scale the
orchestration logic is small enough to own completely, and 'I own every line'
matters when the CTO does code review. Third, the abstractions frameworks give
you — agents, roles, crews — obscure what's happening at the protocol level.
The orchestrator is just a Claude model whose tools are functions that wrap
calls to other Claude models. Once you see it that way, frameworks look like
ceremony around something simple."

## CORPUS INGESTION PIPELINE

### Doc-type-aware chunking strategies

**AI Version of Restore Your Energy** (long narrative, voice sample):
- Section-based chunking using existing markdown headings as boundaries
- ~500–1500 tokens per chunk; sub-split paragraphs if section exceeds 2000 tokens
- `content_type` in {"principle", "concept_explainer", "personal_narrative"}
- `is_voice_anchor: true` on most distinctive Harry-voice passages

**Differentiation + Origin Story** (short marketing docs):
- Heading + paragraph-based chunking, ~300–600 words per chunk
- Origin Story's "For AI Only" Peter Fraser note becomes a discrete chunk
  with `do_not_discuss_flags: ["peter_fraser_death"]`

**Research Summary** (list of studies):
- One chunk per study (delimited by underscores)
- `content_type: "research_finding"`, `has_research_claim: true`
- `quantitative_claim`, `sample_size`, `study_type` as structured fields

**miHealth + BWS product summaries** (structured):
- Section-based chunking
- ER list (1–70) and MR list (1–10): each entry is a row keyed to body system/function
- `content_type: "product_spec"`; ERs/MRs flagged `audience: "practitioner"`

### Annotation pipeline

1. **Chunker** produces raw chunks per doc-type strategies
2. **Gemini embedder** batches chunks for 3072-dim embeddings
3. **LLM annotator** (Claude) proposes metadata per chunk with reasoning
4. **Human review** of `corpus_annotations.yaml`:
   - Explicit-in-source flags manually verified (cannot be delegated)
   - Obvious-from-doc-type annotations accepted
   - Ambiguous cases reviewed and either accepted or overridden
5. **Every annotation carries `annotation_source`:**
   `"explicit_in_source"` | `"obvious_from_doc_type"` |
   `"llm_proposed_accepted"` | `"llm_proposed_overridden"`
6. **Final ingestion script** loads annotations and commits to sqlite-vec
7. **Voice anchor extraction pass:** flags passages exemplifying Harry's
   structural patterns

### Structural voice patterns (extracted from AI Version doc)

Used in Writer system prompts and Validator voice check:
- Open with personal narrative or physics observation
- Introduce concept via everyday analogy (tuning forks, batteries, gel vs liquid)
- Cite specific scientist by name (Pollack, Popp, Lipton, Sheldrake, Szent-Györgyi)
- Land takeaway in italicized one-liner
- Reference numbered Principles of Bioenergetics when relevant

## DATA MODEL

### Source Corpus (sqlite-vec)

| Field                | Type    | Purpose                                                            |
|----------------------|---------|--------------------------------------------------------------------|
| chunk_id             | TEXT PK | Stable identifier — used in citations                              |
| doc_name             | TEXT    | Filename of origin doc                                             |
| doc_type             | TEXT    | Doc-level category used by chunker                                 |
| content              | TEXT    | The chunk text                                                     |
| embedding            | VECTOR  | Gemini 3072-dim (vec0 virtual table, separate row)                 |
| audience_tags        | JSON    | ["consumer"] / ["practitioner"] / ["consumer","practitioner"]      |
| content_type         | TEXT    | "principle" / "concept_explainer" / "personal_narrative" /        |
|                      |         | "product_spec" / "research_finding" / "differentiator" /          |
|                      |         | "origin_story"                                                     |
| product_associations | JSON    | ["miHealth", "BWS", "Infoceuticals", "GEM"]                        |
| do_not_discuss       | BOOL    | True if chunk must not be volunteered or cited                     |
| do_not_discuss_mode  | TEXT    | "never_in_generated_content" / "citation_only" / null             |
| is_voice_anchor      | BOOL    | Strong Harry-voice exemplar                                        |
| corpus_conflict      | TEXT    | Conflict note if chunk contradicts another, else null              |
| annotation_source    | TEXT    | "explicit_in_source" / "obvious_from_doc_type" /                   |
|                      |         | "llm_proposed_accepted" / "llm_proposed_overridden"                |

### ContentBrief (Orchestrator → Writer)

```json
{
  "topic": "string",
  "key_messages": ["string"],
  "audience": "consumer | practitioner | both",
  "funnel_stage": "cold | warm | hot | customer",
  "topic_focus": "fatigue | pain | stress | sleep | cognition | general",
  "product_focus": "infoceuticals | miHealth | BWS | GEM | none_specific",
  "format_intent": "thought_leadership | hook_post | newsletter | sales_email | welcome",
  "tone_register": "physics_first | personal_narrative | research_led | story_led",
  "platform": "linkedin | email",
  "must_include_citations": ["chunk_id"],
  "must_avoid_chunks": ["chunk_id"],
  "trend_context": "string — Researcher output",
  "voice_anchors": ["chunk_id"],
  "vocabulary_register": "lay | clinical"
}
```

### Draft Artifact (Writer → Validator)

```json
{
  "platform": "linkedin | email",
  "fields": {
    "linkedin": { "body": "string" },
    "email":    { "subject": "string", "body": "string" }
  },
  "citations": [
    {
      "claim_span": "exact substring in body",
      "chunk_id": "string",
      "claim_type": "A | B | C"
    }
  ],
  "draft_version": 1
}
```

### Validator Verdict (Validator → Orchestrator)

```json
{
  "verdict": "pass | fail",
  "checks": {
    "citations_resolve":  { "passed": true, "issues": [] },
    "grounding":          { "passed": true, "issues": [] },
    "do_not_discuss":     { "passed": true, "issues": [] },
    "voice":              { "passed": true, "issues": [] },
    "tone":               { "passed": true, "issues": [] }
  },
  "revision_notes": "string — only when verdict == fail"
}
```

### Claim Taxonomy (Validator enforces)

| Type | Description                                           | Action             |
|------|-------------------------------------------------------|--------------------|
| A    | Direct paraphrase of source — must cite               | PASS               |
| B    | Implied by source but not stated — must cite          | PASS WITH FLAG     |
| C    | General knowledge, true, not E4L-specific             | PASS, no cite req  |
| D    | Novel claim not in source                             | HARD FAIL          |

### Observability Schema (sqlite)

```sql
runs(trace_id, brief_json, status, start_ts, end_ts,
     total_cost_usd, turns_used, is_replay, replayed_from)

agent_calls(id, trace_id, agent_name, model,
            tokens_in, tokens_out, cost_usd,
            latency_ms, tool_calls, error, ts)

tool_call_events(id, trace_id, agent_name, tool_name,
                 input_json, output_json, ts)
```

## API / INTERFACE DESIGN

### Backend endpoints

| Method | Path                          | Purpose                                                |
|--------|-------------------------------|--------------------------------------------------------|
| POST   | /generate                     | `{goal, platforms, brief_overrides}` → `{trace_id}`    |
| GET    | /stream/{trace_id}            | SSE: pipeline events as they happen                    |
| GET    | /result/{trace_id}            | Final content pack with citations                      |
| GET    | /runs                         | List recent runs (replay browser)                      |
| GET    | /runs/{trace_id}/detail       | Full run trace: every tool call + message              |
| GET    | /corpus/chunk/{chunk_id}      | Source chunk content (citation hover/click)            |
| GET    | /runs/{trace_id}/replay       | Re-execute same inputs (reproducibility demo)          |

### SSE event types

`run_started`, `research_started`, `research_action`, `research_complete`,
`retrieval_complete`, `brief_ready`, `draft_started`, `draft_ready`,
`validation_started`, `validation_result`, `revision_requested`,
`pack_complete`, `cost_update`, `turn_cap_hit`, `cost_cap_hit`, `run_failed`

### Orchestrator's tools (Anthropic tool schemas)

1. `call_researcher(query) → ResearchSummary`
   — invokes the Researcher mini-agent; corpus retrieval + brief assembly done Python-side on first call_writer
2. `call_writer(platform, brief) → Draft`
   — parallel-callable; both platforms dispatched in one turn via asyncio.gather
3. `call_validator(draft, brief) → ValidatorVerdict`
4. `finalize(status, reason, drafts) → ContentPack`

### Researcher's internal tools (its own loop)

1. `search_web(query) → List[SearchResult]` — Brave Search API
2. `read_url(url) → str` — fetch + clean extraction
3. `score_relevance_to_corpus(text) → float`
   — semantic similarity to source corpus

## OPERATIONAL CONCERNS

### Per-agent configuration

| Agent        | Model        | Temperature | Context discipline                              |
|--------------|--------------|-------------|-------------------------------------------------|
| Orchestrator | Haiku 4.5    | 0.3         | Chunk IDs + summaries only; not full chunks. Sonnet 4.6 fallback if planning degrades (log reason). |
| Researcher   | Haiku 4.5    | 0.5         | Search results + extracted text. Corpus-relevance is cosine, not LLM. |
| Writer       | Sonnet 4.6   | 0.7         | Full cited chunks + voice anchors + brief       |
| Validator    | Opus 4.7     | 0.0         | Draft + cited chunks + editorial rules          |

[RESOLVED 2026-05-15] Opus 4.7 locked for Validator. Empirical comparison run
deferred to S3 (same draft, both models, certainty-inflation + editorial-gate
cases). Orchestrator + Researcher downgraded to Haiku 4.5 — hard logic is in
Python, not the model. Details: DECISIONS.md 2026-05-15 per-agent-model-tiering.

### Rate limiting and cost control

- Exponential backoff on 429 for Anthropic / Gemini / Brave clients
- Per-run cost cap: $0.50
- Orchestrator checks budget before each tool call; clean abort with
  partial result + explanation if cap hits
- Researcher cost cap: $0.10 of per-run budget
- Orchestrator turn cap: 15
- Researcher action cap: 5
- Writer revision cap: 3 rounds per draft
- API endpoint rate limit: 5 requests/min per IP

### Security

- API keys via env vars only — not in client bundle, not logged
- Input sanitization: user `goal` treated as data in Orchestrator's prompt
- No tool that executes arbitrary user input as code
- CORS limited to localhost for the prototype

### Context management discipline

- Orchestrator: chunk IDs + 1-line summaries only, never full chunks
- Writer: cited chunks (top 5–10) + voice anchors (2–3) + brief
- Validator: draft + cited chunks only
- Researcher: search results + extracted text, never full corpus

## DEPLOYMENT

### Two paths, both documented in README

**Quick path:**
```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY, GEMINI_API_KEY, BRAVE_API_KEY
python scripts/ingest_corpus.py source_docs/
uvicorn app.main:app --reload
# open http://localhost:8000
```

**Reproducible path:**
```bash
cp .env.example .env
docker compose up
```

Loom video (3–5 min) walks the system for evaluators who prefer not to
set up API keys locally.

## WHAT CLAUDE CODE SHOULD STRESS-TEST (Phase 1, Opus 4.7)

1. **Orchestrator autonomy budget enforcement**: how are turn and cost checked
   between tool calls without bloating the system prompt? Test with deliberately-
   failing Validator to confirm clean abort.

2. **Citation-resolution programmatic gate**: deterministic string match,
   not LLM judgment. Where exactly does it sit — pre-Validator filter, or
   inside Validator agent? Recommend pre-filter.

3. **Concurrent Writer dispatch**: confirm Anthropic parallel tool calls
   actually parallelize at the wire level, not just in Python event loop.

4. **Researcher corpus-relevance scoring**: how computed (cosine similarity
   to top-K chunks? LLM judge?), and when does it trigger abandoning a
   search direction?

5. **Voice extraction pass during ingestion**: is structural pattern extraction
   reliable without human review pass?

6. **do_not_discuss enforcement**: exact query mechanism at validation time
   given per-chunk metadata.

7. **Brave Search empty/poor results**: graceful degradation path — Researcher
   returns "no trend context available," Orchestrator proceeds without it.

8. **Opus 4.7 vs Sonnet 4.6 for Validator**: small comparison run. Same draft,
   same brief, both models — does Opus catch claim-taxonomy edge cases
   Sonnet misses?

9. **Adversarial input shaping**: define the staged adversarial scenario that
   reliably triggers Validator refusal during demo.

10. **Replay reproducibility contract**: temperature != 0 means runs aren't
    bit-identical. What does "replay" actually guarantee?

## [RESOLVED] ITEMS FROM PRE-PHASE-1 REVIEW

All 10 stress-test items and 13 project-spec open questions resolved in Phase 1
(2026-05-15, Opus 4.7). Key architectural changes vs original spec:

- Model tiering: Orchestrator + Researcher → Haiku 4.5 (was Sonnet 4.6)
- Validator: 5-layer pipeline with deterministic pre-gate (Layer 0) added —
  not in original spec. Load-bearing: corpus itself contains prohibited claims.
- do_not_discuss: mode-aware typed struct + dual enforcement (citation join +
  draft-body scan) — original was a boolean flag only.
- Concurrency: asyncio.gather over two Messages API calls — original spec
  incorrectly stated "Anthropic parallel tool calls" handles it.
- Corpus count: 6 docs (not 7 — was a miscount).

Full resolution notes: project-spec.md "OPEN QUESTIONS — RESOLVED" section.
Full rationale: DECISIONS.md (2026-05-15 entries) + METHODOLOGY.md.
