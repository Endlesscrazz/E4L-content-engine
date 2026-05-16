"""
Tests for the Writer agent.

Unit tests: coercion functions + _parse_draft (deterministic, no API calls).
Integration tests: run_writer with fully mocked Anthropic API responses.

No real API calls — all Anthropic responses are pre-scripted fixtures.
"""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.writer import (
    WRITER_MODEL,
    _coerce_claim,
    _coerce_list,
    _parse_draft,
    run_writer,
)
from app.models import (
    Audience,
    ClaimTaxonomy,
    ContentBrief,
    FunnelStage,
    Platform,
    ResearchResult,
    SourceChunk,
    ToneRegister,
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
def corpus_chunks() -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id="mihealth_001",
            doc_name="miHealth Product Doc",
            doc_type="product",
            content="The miHealth uses PEMF and microcurrent to support cellular energy.",
        )
    ]


@pytest.fixture
def voice_anchors() -> list[SourceChunk]:
    return [
        SourceChunk(
            chunk_id="voice_001",
            doc_name="Origin Story",
            doc_type="origin_story",
            content="I discovered that the body has an information field that governs health.",
            is_voice_anchor=True,
        )
    ]


def _make_tool_block(tool_input: dict[str, Any]) -> MagicMock:
    """Build a mock tool_use block as the Anthropic SDK would return it."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "submit_draft"
    block.input = tool_input
    return block


def _make_response(tool_input: dict[str, Any]) -> tuple[MagicMock, float]:
    """Build a mock (Message, cost) tuple."""
    msg = MagicMock()
    msg.stop_reason = "tool_use"
    msg.content = [_make_tool_block(tool_input)]
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    return msg, 0.002


# ─── _coerce_list ──────────────────────────────────────────────────────────────

def test_coerce_list_passthrough():
    assert _coerce_list(["a", "b"]) == ["a", "b"]


def test_coerce_list_json_string():
    assert _coerce_list('["a", "b"]') == ["a", "b"]


def test_coerce_list_empty_string():
    assert _coerce_list("") == []


def test_coerce_list_none():
    assert _coerce_list(None) == []


def test_coerce_list_plain_string():
    result = _coerce_list("some item")
    assert result == ["some item"]


def test_coerce_list_nested_objects():
    raw = '[{"text": "claim1", "taxonomy": "A"}]'
    result = _coerce_list(raw)
    assert len(result) == 1
    assert result[0]["text"] == "claim1"


# ─── _coerce_claim ─────────────────────────────────────────────────────────────

def test_coerce_claim_dict_passthrough():
    d = {"text": "x", "taxonomy": "A"}
    assert _coerce_claim(d) is d


def test_coerce_claim_json_string():
    s = '{"text": "x", "taxonomy": "A", "certainty_inflation": false}'
    result = _coerce_claim(s)
    assert isinstance(result, dict)
    assert result["text"] == "x"


def test_coerce_claim_invalid_string():
    result = _coerce_claim("not json")
    assert result == "not json"  # returned as-is; _parse_draft logs and skips


def test_coerce_claim_empty_string():
    result = _coerce_claim("")
    assert result == ""


# ─── _parse_draft ──────────────────────────────────────────────────────────────

def test_parse_draft_linkedin():
    tool_input = {
        "body": "The miHealth supports cellular vitality through PEMF.",
        "claims": [
            {
                "text": "The miHealth supports cellular vitality through PEMF.",
                "chunk_id": "mihealth_001",
                "cited_substring": "PEMF and microcurrent to support cellular energy",
                "taxonomy": "A",
                "certainty_inflation": False,
            }
        ],
        "cited_chunk_ids": ["mihealth_001"],
    }
    draft = _parse_draft(tool_input, Platform.LINKEDIN)
    assert draft is not None
    assert draft.platform == Platform.LINKEDIN
    assert draft.subject is None
    assert len(draft.claims) == 1
    assert draft.claims[0].taxonomy == ClaimTaxonomy.A
    assert draft.cited_chunk_ids == ["mihealth_001"]


def test_parse_draft_email_with_subject():
    tool_input = {
        "subject": "Why your energy crash isn't a mystery",
        "body": "Your body has an information field. When it loses coherence, fatigue follows.",
        "claims": [],
        "cited_chunk_ids": [],
    }
    draft = _parse_draft(tool_input, Platform.EMAIL)
    assert draft is not None
    assert draft.platform == Platform.EMAIL
    assert draft.subject == "Why your energy crash isn't a mystery"


def test_parse_draft_coerces_claims_array():
    """Sonnet sometimes returns claims as a JSON-encoded string."""
    claims_as_string = json.dumps([
        {"text": "claim", "taxonomy": "C", "certainty_inflation": False}
    ])
    tool_input = {
        "body": "Some body text.",
        "claims": claims_as_string,
        "cited_chunk_ids": [],
    }
    draft = _parse_draft(tool_input, Platform.LINKEDIN)
    assert draft is not None
    assert len(draft.claims) == 1
    assert draft.claims[0].text == "claim"


def test_parse_draft_coerces_individual_claim_objects():
    """Sonnet sometimes stringifies individual claim dicts within the array."""
    claim_as_string = json.dumps({"text": "nested claim", "taxonomy": "B", "certainty_inflation": True})
    tool_input = {
        "body": "Body text.",
        "claims": [claim_as_string],
        "cited_chunk_ids": [],
    }
    draft = _parse_draft(tool_input, Platform.LINKEDIN)
    assert draft is not None
    assert len(draft.claims) == 1
    assert draft.claims[0].text == "nested claim"
    assert draft.claims[0].taxonomy == ClaimTaxonomy.B


def test_parse_draft_skips_malformed_claims():
    """Malformed claims are logged and skipped — parse still succeeds."""
    tool_input = {
        "body": "Body text.",
        "claims": [
            {"text": "good claim", "taxonomy": "C", "certainty_inflation": False},
            "not a dict and not valid json",
            None,
        ],
        "cited_chunk_ids": [],
    }
    draft = _parse_draft(tool_input, Platform.LINKEDIN)
    assert draft is not None
    assert len(draft.claims) == 1  # only the valid one


def test_parse_draft_missing_body_returns_none():
    draft = _parse_draft({"claims": [], "cited_chunk_ids": []}, Platform.LINKEDIN)
    assert draft is None


def test_parse_draft_coerces_cited_chunk_ids():
    """cited_chunk_ids may also be stringified."""
    tool_input = {
        "body": "Body.",
        "claims": [],
        "cited_chunk_ids": '["chunk_001", "chunk_002"]',
    }
    draft = _parse_draft(tool_input, Platform.LINKEDIN)
    assert draft is not None
    assert draft.cited_chunk_ids == ["chunk_001", "chunk_002"]


# ─── run_writer integration (mocked API) ─────────────────────────────────────

@pytest.fixture
def mock_client():
    return MagicMock()


def test_run_writer_linkedin_success(brief, corpus_chunks, voice_anchors, mock_client):
    """Happy path: API returns valid submit_draft → Draft returned."""
    tool_input = {
        "body": "Discover how the miHealth device supports your body field.",
        "claims": [
            {
                "text": "miHealth device supports your body field",
                "chunk_id": "mihealth_001",
                "cited_substring": "PEMF and microcurrent to support cellular energy",
                "taxonomy": "A",
                "certainty_inflation": False,
            }
        ],
        "cited_chunk_ids": ["mihealth_001"],
    }
    mock_client.create.return_value = _make_response(tool_input)

    draft = asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_001",
    ))

    assert draft is not None
    assert draft.platform == Platform.LINKEDIN
    assert draft.model_used == WRITER_MODEL
    assert draft.cost_usd == 0.002
    assert len(draft.claims) == 1
    mock_client.create.assert_called_once()


def test_run_writer_email_success(brief, corpus_chunks, voice_anchors, mock_client):
    """Email path: subject field required and returned."""
    tool_input = {
        "subject": "The hidden reason your energy crashes",
        "body": "Most fatigue isn't about sleep. It's about your body's energy field.",
        "claims": [],
        "cited_chunk_ids": [],
    }
    mock_client.create.return_value = _make_response(tool_input)

    draft = asyncio.run(run_writer(
        platform=Platform.EMAIL,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_002",
    ))

    assert draft is not None
    assert draft.platform == Platform.EMAIL
    assert draft.subject == "The hidden reason your energy crashes"


def test_run_writer_with_revision_notes(brief, corpus_chunks, voice_anchors, mock_client):
    """Revision notes are included in the user message (verified via call args)."""
    tool_input = {
        "body": "Revised body text addressing validator feedback.",
        "claims": [],
        "cited_chunk_ids": [],
    }
    mock_client.create.return_value = _make_response(tool_input)

    asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes="Remove the claim that miHealth cures fatigue.",
        anthropic_client=mock_client,
        trace_id="test_003",
    ))

    call_kwargs = mock_client.create.call_args
    assert call_kwargs is not None
    # asyncio.to_thread passes all args as kwargs — use .kwargs directly
    messages = call_kwargs.kwargs.get("messages", [])
    assert messages, "messages not found in call kwargs"
    user_content = messages[0]["content"]
    assert "REVISION NOTES" in user_content
    assert "Remove the claim" in user_content


def test_run_writer_no_submit_draft_returns_none(brief, corpus_chunks, voice_anchors, mock_client):
    """If API doesn't call submit_draft, run_writer returns None."""
    msg = MagicMock()
    msg.stop_reason = "end_turn"
    msg.content = []
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    mock_client.create.return_value = (msg, 0.001)

    draft = asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_004",
    ))

    assert draft is None


def test_run_writer_parse_failure_returns_none(brief, corpus_chunks, voice_anchors, mock_client):
    """If _parse_draft fails (missing required body), returns None."""
    # Missing 'body' key — parse will fail
    bad_input = {"claims": [], "cited_chunk_ids": []}
    mock_client.create.return_value = _make_response(bad_input)

    draft = asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_005",
    ))

    assert draft is None


def test_run_writer_with_research_context(brief, corpus_chunks, voice_anchors, mock_client):
    """Research findings are included in the user message."""
    from app.models import ResearchFinding
    research = ResearchResult(
        findings=[
            ResearchFinding(
                title="Mitochondrial health trends 2025",
                url="https://example.com/article",
                snippet="New research shows mitochondrial function supports energy metabolism.",
                relevance_score=0.72,
                relevance_label="keep",
            )
        ]
    )

    tool_input = {"body": "Body.", "claims": [], "cited_chunk_ids": []}
    mock_client.create.return_value = _make_response(tool_input)

    asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=research,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_006",
    ))

    call_kwargs = mock_client.create.call_args
    assert call_kwargs is not None
    messages = call_kwargs.kwargs.get("messages", [])
    assert messages
    user_content = messages[0]["content"]
    assert "EXTERNAL RESEARCH CONTEXT" in user_content
    assert "Mitochondrial health trends 2025" in user_content


def test_run_writer_api_error_returns_none(brief, corpus_chunks, voice_anchors, mock_client):
    """API exception returns None (does not raise)."""
    mock_client.create.side_effect = RuntimeError("API unavailable")

    draft = asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_007",
    ))

    assert draft is None


def test_run_writer_on_event_called(brief, corpus_chunks, voice_anchors, mock_client):
    """on_event callback is called for start and done events."""
    tool_input = {"body": "Body.", "claims": [], "cited_chunk_ids": []}
    mock_client.create.return_value = _make_response(tool_input)

    events: list[tuple[str, dict]] = []

    def capture(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_008",
        on_event=capture,
    ))

    event_types = [e[0] for e in events]
    assert "writer_start" in event_types
    assert "writer_done" in event_types


def test_run_writer_forced_tool_choice(brief, corpus_chunks, voice_anchors, mock_client):
    """Verifies tool_choice={"type":"tool","name":"submit_draft"} is passed."""
    tool_input = {"body": "Body.", "claims": [], "cited_chunk_ids": []}
    mock_client.create.return_value = _make_response(tool_input)

    asyncio.run(run_writer(
        platform=Platform.LINKEDIN,
        brief=brief,
        corpus_chunks=corpus_chunks,
        voice_anchors=voice_anchors,
        research=None,
        revision_notes=None,
        anthropic_client=mock_client,
        trace_id="test_009",
    ))

    call_kwargs = mock_client.create.call_args
    tool_choice = call_kwargs.kwargs.get("tool_choice")
    assert tool_choice is not None
    assert tool_choice["type"] == "tool"
    assert tool_choice["name"] == "submit_draft"
