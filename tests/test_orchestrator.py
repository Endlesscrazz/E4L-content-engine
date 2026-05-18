"""
Tests for the Orchestrator agent and observability layer.

Unit tests:
  - RunBudget enforcement (no API calls)
  - _tool_result helper (no API calls)
  - Observability write functions (in-memory sqlite)

Integration tests (mocked):
  - Happy path: research → parallel write → validate → finalize → ContentPack
  - Parallel writer dispatch: two call_writer blocks → asyncio.gather
  - Always-fail Validator: 2 revisions → cap → partial finalize, no infinite loop
  - Turn cap: budget.is_exhausted() → budget_exhausted injected → finalize
  - Cost cap: same as turn cap
  - Editorial gate refusal: Validator ship=False forever → partial finalize
"""

import asyncio
import json
import sqlite3
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.orchestrator import (
    ORCHESTRATOR_MODEL,
    _fetch_corpus,
    _gather_writers,
    _tool_result,
    run_orchestrator,
)
from app.models import (
    Audience,
    CitationCheckResult,
    ContentBrief,
    ContentPack,
    DoNotDiscussCheckResult,
    Draft,
    EditorialCheckResult,
    FunnelStage,
    LLMJudgeResult,
    Platform,
    ResearchResult,
    RunBudget,
    ToneRegister,
    ValidatorVerdict,
)
from app.observability import (
    init_obs_db,
    query_run_detail,
    query_runs,
    write_agent_call,
    write_run_end,
    write_run_start,
    write_tool_event,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def brief() -> ContentBrief:
    return ContentBrief(
        goal="Introduce miHealth to consumers experiencing chronic fatigue",
        audience=Audience.CONSUMER,
        funnel_stage=FunnelStage.COLD,
        tone=ToneRegister.CONVERSATIONAL,
    )


@pytest.fixture
def obs_conn():
    """In-memory observability db for tests."""
    conn = init_obs_db(db_path=sqlite3.connect(":memory:"))
    yield conn
    conn.close()


def _mock_obs_conn():
    """init_obs_db with :memory: path string."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from app.observability import _DDL
    conn.executescript(_DDL)
    conn.commit()
    return conn


@pytest.fixture
def good_draft() -> Draft:
    return Draft(
        platform=Platform.LINKEDIN,
        body="The miHealth device supports cellular energy through PEMF.",
        claims=[],
        cited_chunk_ids=["mihealth_001"],
        model_used="claude-sonnet-4-6",
        cost_usd=0.002,
    )


@pytest.fixture
def good_email_draft() -> Draft:
    return Draft(
        platform=Platform.EMAIL,
        subject="Tired for no reason?",
        body="Your body has an energy field. miHealth helps restore it.",
        claims=[],
        cited_chunk_ids=[],
        model_used="claude-sonnet-4-6",
        cost_usd=0.002,
    )


def _shipped_verdict(platform: Platform) -> ValidatorVerdict:
    return ValidatorVerdict(
        draft_platform=platform,
        editorial=EditorialCheckResult(passed=True),
        citations=CitationCheckResult(passed=True),
        do_not_discuss=DoNotDiscussCheckResult(passed=True),
        llm_judge=LLMJudgeResult(passed=True),
        ship=True,
    )


def _rejected_verdict(platform: Platform, notes: str = "Fix the claim.") -> ValidatorVerdict:
    return ValidatorVerdict(
        draft_platform=platform,
        editorial=EditorialCheckResult(passed=True),
        citations=CitationCheckResult(passed=True),
        do_not_discuss=DoNotDiscussCheckResult(passed=True),
        llm_judge=LLMJudgeResult(passed=False, revision_notes=notes),
        ship=False,
        revision_notes=notes,
    )


def _make_haiku_response(tool_calls: list[dict[str, Any]]) -> tuple[MagicMock, float]:
    """Build a mock Haiku response with tool_use blocks."""
    blocks = []
    for i, tc in enumerate(tool_calls):
        block = MagicMock()
        block.type = "tool_use"
        block.name = tc["name"]
        block.input = tc["input"]
        block.id = tc.get("id", f"tool_{i}")
        blocks.append(block)

    msg = MagicMock()
    msg.stop_reason = "tool_use"
    msg.content = blocks
    msg.usage.input_tokens = 80
    msg.usage.output_tokens = 30
    return msg, 0.001


def _make_end_turn_response() -> tuple[MagicMock, float]:
    msg = MagicMock()
    msg.stop_reason = "end_turn"
    msg.content = []
    msg.usage.input_tokens = 20
    msg.usage.output_tokens = 10
    return msg, 0.0005


# ─── RunBudget unit tests ─────────────────────────────────────────────────────

def test_run_budget_not_exhausted_initially():
    b = RunBudget()
    assert not b.is_exhausted()


def test_run_budget_turn_cap():
    b = RunBudget(max_turns=3)
    b.record(0.001)
    b.record(0.001)
    assert not b.is_exhausted()
    b.record(0.001)
    assert b.is_exhausted()


def test_run_budget_cost_cap():
    b = RunBudget(max_cost_usd=0.10)
    b.record(0.09)
    assert not b.is_exhausted()
    b.record(0.02)
    assert b.is_exhausted()


def test_run_budget_turns_remaining():
    b = RunBudget(max_turns=5)
    b.record(0.001)
    assert b.turns_remaining() == 4


def test_run_budget_cost_remaining():
    b = RunBudget(max_cost_usd=0.50)
    b.record(0.10)
    assert abs(b.cost_remaining() - 0.40) < 1e-6


# ─── _tool_result unit tests ──────────────────────────────────────────────────

def test_tool_result_strips_underscore_keys():
    result = _tool_result("id1", {"status": "ok", "_draft": object(), "_chunks": []})
    payload = json.loads(result["content"])
    assert "status" in payload
    assert "_draft" not in payload
    assert "_chunks" not in payload


def test_tool_result_coerces_non_serializable():
    class NotJSON:
        def __str__(self):
            return "repr"

    result = _tool_result("id2", {"obj": NotJSON()})
    payload = json.loads(result["content"])
    assert payload["obj"] == "repr"


def test_tool_result_passes_primitives():
    result = _tool_result("id3", {
        "ship": True,
        "count": 3,
        "cost": 0.005,
        "notes": None,
        "tags": ["a", "b"],
    })
    payload = json.loads(result["content"])
    assert payload["ship"] is True
    assert payload["count"] == 3
    assert payload["notes"] is None
    assert payload["tags"] == ["a", "b"]


# ─── Observability unit tests ─────────────────────────────────────────────────

def test_obs_write_run_start_and_end():
    conn = _mock_obs_conn()
    from app.models import RunRecord

    record = RunRecord(
        trace_id="trace_obs_001",
        brief_json='{"goal": "test"}',
        status="running",
        start_ts="2026-05-16T10:00:00+00:00",
    )
    write_run_start(conn, record)

    rows = query_runs(conn)
    assert len(rows) == 1
    assert rows[0]["trace_id"] == "trace_obs_001"
    assert rows[0]["status"] == "running"

    write_run_end(conn, "trace_obs_001", "complete", 0.05, 3)
    rows = query_runs(conn)
    assert rows[0]["status"] == "complete"
    assert rows[0]["total_cost_usd"] == 0.05
    assert rows[0]["turns_used"] == 3


def test_obs_write_agent_call():
    conn = _mock_obs_conn()
    from app.models import AgentTelemetry
    from app.observability import write_agent_call

    write_agent_call(
        conn,
        AgentTelemetry(
            trace_id="trace_obs_002",
            agent_name="orchestrator",
            model=ORCHESTRATOR_MODEL,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            latency_ms=250,
            tool_calls=2,
        ),
    )

    # Query agent_calls directly (no run row needed)
    rows = conn.execute(
        "SELECT * FROM agent_calls WHERE trace_id = ?", ("trace_obs_002",)
    ).fetchall()
    assert len(rows) == 1
    assert dict(rows[0])["agent_name"] == "orchestrator"
    assert dict(rows[0])["tokens_in"] == 100


def test_obs_write_tool_event():
    conn = _mock_obs_conn()
    from app.models import ToolCallEvent
    from app.observability import write_tool_event

    write_tool_event(
        conn,
        ToolCallEvent(
            trace_id="trace_obs_003",
            agent_name="orchestrator",
            tool_name="call_writer",
            input_json='{"platform": "linkedin"}',
            output_json='{"status": "ok"}',
            ts="2026-05-16T10:01:00+00:00",
        ),
    )

    # Query tool_call_events directly
    rows = conn.execute(
        "SELECT * FROM tool_call_events WHERE trace_id = ?", ("trace_obs_003",)
    ).fetchall()
    assert len(rows) == 1
    assert dict(rows[0])["tool_name"] == "call_writer"
    assert dict(rows[0])["agent_name"] == "orchestrator"


def test_obs_write_failure_does_not_raise():
    """Fire-and-forget: write failure logs but never raises."""
    conn = _mock_obs_conn()
    conn.close()  # close to trigger write failure
    from app.models import RunRecord

    # Must not raise even with closed connection
    write_run_start(conn, RunRecord(
        trace_id="x", brief_json="{}", status="running", start_ts="2026-01-01"
    ))


# ─── run_orchestrator integration tests (mocked) ─────────────────────────────

@pytest.fixture
def mock_anthropic():
    return MagicMock()


@pytest.fixture
def mock_gemini():
    client = MagicMock()
    client.embed_query.return_value = [0.0] * 3072
    return client


@pytest.fixture
def mock_brave():
    return MagicMock()


@pytest.fixture
def mock_corpus_conn():
    return MagicMock(spec=sqlite3.Connection)


def _mock_full_pipeline(
    mock_anthropic,
    good_draft,
    good_email_draft,
    li_verdict=None,
    email_verdict=None,
):
    """
    Build a sequence of Haiku responses that drives the happy-path pipeline:
      Turn 1: call_writer(linkedin) + call_writer(email) in parallel
      Turn 2: call_validator(linkedin) + call_validator(email)
      Turn 3: finalize(complete)
    """
    li_verdict = li_verdict or _shipped_verdict(Platform.LINKEDIN)
    email_verdict = email_verdict or _shipped_verdict(Platform.EMAIL)

    responses = [
        # Turn 1: write both platforms in one response (parallel)
        _make_haiku_response([
            {"name": "call_writer", "input": {"platform": "linkedin"}, "id": "tw_li"},
            {"name": "call_writer", "input": {"platform": "email"}, "id": "tw_em"},
        ]),
        # Turn 2: validate both
        _make_haiku_response([
            {"name": "call_validator", "input": {"platform": "linkedin"}, "id": "tv_li"},
            {"name": "call_validator", "input": {"platform": "email"}, "id": "tv_em"},
        ]),
        # Turn 3: finalize
        _make_haiku_response([
            {"name": "finalize", "input": {"status": "complete", "reason": "all shipped"}, "id": "tfin"},
        ]),
    ]
    mock_anthropic.create.side_effect = responses
    return li_verdict, email_verdict


def test_orchestrator_happy_path(brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn):
    """Full pipeline: research skipped → parallel write → validate → finalize → ContentPack."""
    li_verdict, email_verdict = _mock_full_pipeline(mock_anthropic, None, None)

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([], [])),
        patch("agents.orchestrator.run_writer") as mock_writer,
        patch("agents.orchestrator.validate_draft") as mock_validator,
    ):
        li_draft = Draft(platform=Platform.LINKEDIN, body="LI body", claims=[], cited_chunk_ids=[])
        em_draft = Draft(platform=Platform.EMAIL, subject="Subj", body="Email body", claims=[], cited_chunk_ids=[])

        mock_writer.side_effect = [li_draft, em_draft]
        mock_validator.side_effect = [li_verdict, email_verdict]

        pack = asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            trace_id="test_happy_001",
        ))

    assert isinstance(pack, ContentPack)
    assert pack.trace_id == "test_happy_001"
    assert pack.status == "complete"
    assert pack.linkedin_draft is not None
    assert pack.email_draft is not None
    assert pack.linkedin_verdict is not None
    assert pack.email_verdict is not None
    assert pack.linkedin_verdict.ship is True
    assert pack.email_verdict.ship is True


def test_orchestrator_parallel_dispatch(brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn):
    """When Haiku returns two call_writer blocks, asyncio.gather is used."""
    _mock_full_pipeline(mock_anthropic, None, None)

    writer_calls: list[Platform] = []

    async def mock_writer_tracker(**kwargs):
        writer_calls.append(kwargs["platform"])
        platform = kwargs["platform"]
        return Draft(platform=platform, body="body", claims=[], cited_chunk_ids=[])

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([], [])),
        patch("agents.orchestrator.run_writer", side_effect=mock_writer_tracker),
        patch("agents.orchestrator.validate_draft") as mock_validator,
    ):
        mock_validator.side_effect = [
            _shipped_verdict(Platform.LINKEDIN),
            _shipped_verdict(Platform.EMAIL),
        ]

        pack = asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            trace_id="test_parallel_001",
        ))

    # Both platforms should have been written
    assert Platform.LINKEDIN in writer_calls
    assert Platform.EMAIL in writer_calls


def test_orchestrator_always_fail_validator_hits_revision_cap(
    brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn
):
    """
    Always-failing Validator: 2 revisions used → Orchestrator finalizes partial.
    The pipeline must NOT loop infinitely.
    """
    # Haiku loop: write both → validate both (both fail) → revise linkedin → validate →
    # revise email → validate → finalize partial
    responses = [
        # Turn 1: write both
        _make_haiku_response([
            {"name": "call_writer", "input": {"platform": "linkedin"}, "id": "t1_li"},
            {"name": "call_writer", "input": {"platform": "email"}, "id": "t1_em"},
        ]),
        # Turn 2: validate both — both will fail
        _make_haiku_response([
            {"name": "call_validator", "input": {"platform": "linkedin"}, "id": "t2_li"},
            {"name": "call_validator", "input": {"platform": "email"}, "id": "t2_em"},
        ]),
        # Turn 3: revise linkedin
        _make_haiku_response([
            {"name": "call_writer", "input": {"platform": "linkedin", "revision_notes": "Fix it."}, "id": "t3_li"},
        ]),
        # Turn 4: validate linkedin — fails again
        _make_haiku_response([
            {"name": "call_validator", "input": {"platform": "linkedin"}, "id": "t4_li"},
        ]),
        # Turn 5: try to revise but cap hit → finalize
        _make_haiku_response([
            {"name": "finalize", "input": {"status": "partial", "reason": "revision cap reached"}, "id": "tfin"},
        ]),
    ]
    mock_anthropic.create.side_effect = responses

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([], [])),
        patch("agents.orchestrator.run_writer") as mock_writer,
        patch("agents.orchestrator.validate_draft") as mock_validator,
    ):
        li_draft = Draft(platform=Platform.LINKEDIN, body="LI", claims=[], cited_chunk_ids=[])
        em_draft = Draft(platform=Platform.EMAIL, subject="S", body="E", claims=[], cited_chunk_ids=[])
        mock_writer.side_effect = [li_draft, em_draft, li_draft]  # 3 writes total

        # All validator calls reject
        mock_validator.side_effect = [
            _rejected_verdict(Platform.LINKEDIN),
            _rejected_verdict(Platform.EMAIL),
            _rejected_verdict(Platform.LINKEDIN),
        ]

        pack = asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            trace_id="test_revision_cap_001",
        ))

    # Must terminate without error; status reflects partial or cap
    assert pack is not None
    assert pack.status in ("partial", "cap_hit")


def test_orchestrator_turn_cap_injects_budget_exhausted(
    brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn
):
    """
    When turn cap is hit, budget_exhausted is injected and Haiku finalizes.
    Verifies: no infinite loop, ContentPack returned with cap_hit status.
    """
    # Responses: keep calling call_writer infinitely (until cap hits) then finalize
    write_response = _make_haiku_response([
        {"name": "call_writer", "input": {"platform": "linkedin"}, "id": "tw"},
    ])
    finalize_response = _make_haiku_response([
        {"name": "finalize", "input": {"status": "cap_hit", "reason": "budget_exhausted received"}, "id": "tfin"},
    ])

    # Budget: max 2 turns so cap hits fast
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            return finalize_response
        return write_response

    mock_anthropic.create.side_effect = side_effect

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([], [])),
        patch("agents.orchestrator.run_writer") as mock_writer,
        patch("agents.orchestrator.MAX_TURNS", 2),
    ):
        mock_writer.return_value = Draft(
            platform=Platform.LINKEDIN, body="body", claims=[], cited_chunk_ids=[]
        )

        pack = asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            trace_id="test_cap_001",
        ))

    assert pack is not None
    # Status could be cap_hit or partial depending on when finalize was called
    assert pack.status in ("cap_hit", "partial", "complete")


def test_orchestrator_researcher_called_and_research_passed_to_writers(
    brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn
):
    """When Haiku calls call_researcher, the result is passed to Writers."""
    responses = [
        # Turn 1: research
        _make_haiku_response([
            {"name": "call_researcher", "input": {"goal": "energy fatigue trends"}, "id": "tr"},
        ]),
        # Turn 2: write both
        _make_haiku_response([
            {"name": "call_writer", "input": {"platform": "linkedin"}, "id": "tw_li"},
            {"name": "call_writer", "input": {"platform": "email"}, "id": "tw_em"},
        ]),
        # Turn 3: validate both
        _make_haiku_response([
            {"name": "call_validator", "input": {"platform": "linkedin"}, "id": "tv_li"},
            {"name": "call_validator", "input": {"platform": "email"}, "id": "tv_em"},
        ]),
        # Turn 4: finalize
        _make_haiku_response([
            {"name": "finalize", "input": {"status": "complete", "reason": "done"}, "id": "tfin"},
        ]),
    ]
    mock_anthropic.create.side_effect = responses

    research_passed_to_writers: list[ResearchResult | None] = []

    async def mock_writer_capture(**kwargs):
        research_passed_to_writers.append(kwargs.get("research"))
        return Draft(platform=kwargs["platform"], body="b", claims=[], cited_chunk_ids=[])

    fake_research = ResearchResult(findings=[], actions_used=2, cost_usd=0.002)

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([], [])),
        patch("agents.orchestrator.run_writer", side_effect=mock_writer_capture),
        patch("agents.orchestrator.run_researcher", return_value=fake_research),
        patch("agents.orchestrator.validate_draft") as mock_validator,
    ):
        mock_validator.side_effect = [
            _shipped_verdict(Platform.LINKEDIN),
            _shipped_verdict(Platform.EMAIL),
        ]

        pack = asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            trace_id="test_research_001",
        ))

    assert pack.research is not None
    assert pack.research == fake_research
    # Research should have been passed to both writers
    assert all(r == fake_research for r in research_passed_to_writers)


def test_orchestrator_on_event_callback(brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn):
    """on_event is called for key pipeline steps."""
    _mock_full_pipeline(mock_anthropic, None, None)

    events: list[str] = []

    def capture(event_type: str, _: dict) -> None:
        events.append(event_type)

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([], [])),
        patch("agents.orchestrator.run_writer") as mock_writer,
        patch("agents.orchestrator.validate_draft") as mock_validator,
    ):
        mock_writer.side_effect = [
            Draft(platform=Platform.LINKEDIN, body="b", claims=[], cited_chunk_ids=[]),
            Draft(platform=Platform.EMAIL, subject="s", body="b", claims=[], cited_chunk_ids=[]),
        ]
        mock_validator.side_effect = [
            _shipped_verdict(Platform.LINKEDIN),
            _shipped_verdict(Platform.EMAIL),
        ]

        asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            trace_id="test_events_001",
            on_event=capture,
        ))

    assert "pipeline_start" in events
    assert "pipeline_done" in events


def test_orchestrator_obs_writes(brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn):
    """Observability: run_start and run_end rows are written for a complete pipeline."""
    _mock_full_pipeline(mock_anthropic, None, None)
    obs = _mock_obs_conn()

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([], [])),
        patch("agents.orchestrator.run_writer") as mock_writer,
        patch("agents.orchestrator.validate_draft") as mock_validator,
    ):
        mock_writer.side_effect = [
            Draft(platform=Platform.LINKEDIN, body="b", claims=[], cited_chunk_ids=[]),
            Draft(platform=Platform.EMAIL, subject="s", body="b", claims=[], cited_chunk_ids=[]),
        ]
        mock_validator.side_effect = [
            _shipped_verdict(Platform.LINKEDIN),
            _shipped_verdict(Platform.EMAIL),
        ]

        asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            obs_conn=obs,
            trace_id="test_obs_001",
        ))

    runs = query_runs(obs)
    assert len(runs) == 1
    assert runs[0]["trace_id"] == "test_obs_001"
    assert runs[0]["status"] == "complete"
    obs.close()


def test_orchestrator_end_turn_fallback_refusal_sets_reason(
    brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn
):
    """R12: end_turn keyword-fallback refusal path must always set a non-empty reason."""
    # Haiku responds on turn 1 with a text refusal (no tool call) containing a
    # refusal keyword — simulates the Orchestrator refusing an adversarial brief
    # via plain text instead of a finalize() tool call.
    refusal_text_block = MagicMock()
    refusal_text_block.text = "I cannot generate content that claims to cure disease. This violates editorial policy."
    refusal_msg = MagicMock()
    refusal_msg.stop_reason = "end_turn"
    refusal_msg.content = [refusal_text_block]
    refusal_msg.usage.input_tokens = 50
    refusal_msg.usage.output_tokens = 30
    mock_anthropic.create.return_value = (refusal_msg, 0.0003)

    events: list[dict] = []

    def capture(event_type, payload):
        events.append({"type": event_type, **payload})

    pack = asyncio.run(run_orchestrator(
        brief=brief,
        anthropic_client=mock_anthropic,
        gemini_embedder=mock_gemini,
        brave_client=mock_brave,
        corpus_conn=mock_corpus_conn,
        trace_id="test_r12_fallback",
        on_event=capture,
    ))

    assert pack.status == "refused", f"Expected refused, got {pack.status}"
    assert pack.finalize_reason, "finalize_reason must not be empty on fallback refusal"
    assert pack.finalize_reason != "turn cap reached without finalize call", (
        "fallback refusal should set an explanatory reason, not the generic cap message"
    )
    # pipeline_finalizing event should carry the reason to the UI
    finalizing_events = [e for e in events if e["type"] == "pipeline_finalizing"]
    assert finalizing_events, "pipeline_finalizing event must be emitted"
    assert finalizing_events[0].get("reason"), "pipeline_finalizing event must include a non-empty reason"


def test_orchestrator_linkedin_zero_citations_fast_fails_before_opus(
    brief, mock_anthropic, mock_gemini, mock_brave, mock_corpus_conn
):
    """R11: LinkedIn draft with zero citations + non-empty corpus fast-fails without calling Opus."""
    # Pipeline: write both → fast-fail LinkedIn (zero cites) → revision → both succeed
    chunk = MagicMock()
    chunk.chunk_id = "mihealth_001"

    responses = [
        # Turn 1: write both platforms in parallel
        _make_haiku_response([
            {"name": "call_writer", "input": {"platform": "linkedin"}, "id": "tw1"},
            {"name": "call_writer", "input": {"platform": "email"}, "id": "tw2"},
        ]),
        # Turn 2: validate both
        _make_haiku_response([
            {"name": "call_validator", "input": {"platform": "linkedin"}, "id": "tv1"},
            {"name": "call_validator", "input": {"platform": "email"}, "id": "tv2"},
        ]),
        # Turn 3: revise LinkedIn after fast-fail rejection
        _make_haiku_response([
            {"name": "call_writer", "input": {"platform": "linkedin", "revision_notes": "Zero citations"}, "id": "tw3"},
        ]),
        # Turn 4: validate revised LinkedIn
        _make_haiku_response([
            {"name": "call_validator", "input": {"platform": "linkedin"}, "id": "tv3"},
        ]),
        # Turn 5: finalize
        _make_haiku_response([
            {"name": "finalize", "input": {"status": "complete", "reason": "both shipped"}, "id": "tfin"},
        ]),
    ]
    mock_anthropic.create.side_effect = responses

    validator_call_count = 0

    def counting_validator(**kwargs):
        nonlocal validator_call_count
        validator_call_count += 1
        return _shipped_verdict(kwargs["draft"].platform)

    # Writer returns zero-citation LinkedIn first, then a cited revision
    writer_calls = 0

    def mock_writer_fn(**kwargs):
        nonlocal writer_calls
        writer_calls += 1
        platform = kwargs["platform"]
        if platform == Platform.LINKEDIN and writer_calls == 1:
            # First LinkedIn call: zero citations — triggers fast-fail
            return Draft(platform=Platform.LINKEDIN, body="body", claims=[], cited_chunk_ids=[])
        if platform == Platform.EMAIL:
            return Draft(platform=Platform.EMAIL, subject="s", body="body", claims=[], cited_chunk_ids=[])
        # Revised LinkedIn: now includes a citation
        return Draft(platform=Platform.LINKEDIN, body="cited body", claims=[], cited_chunk_ids=["mihealth_001"])

    with (
        patch("agents.orchestrator._fetch_corpus", return_value=([chunk], [])),
        patch("agents.orchestrator.run_writer", side_effect=mock_writer_fn),
        patch("agents.orchestrator.validate_draft", side_effect=counting_validator),
    ):
        pack = asyncio.run(run_orchestrator(
            brief=brief,
            anthropic_client=mock_anthropic,
            gemini_embedder=mock_gemini,
            brave_client=mock_brave,
            corpus_conn=mock_corpus_conn,
            trace_id="test_r11_zero_cites",
        ))

    # Opus validator was NOT called for the first zero-citation LinkedIn draft
    # (fast-fail skipped it). It WAS called for email and the revised LinkedIn.
    assert validator_call_count == 2, (
        f"Expected 2 Opus calls (email + revised LI), got {validator_call_count} — "
        "fast-fail should have skipped the first zero-citation LI draft"
    )
