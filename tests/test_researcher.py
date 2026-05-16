"""
Tests for the Researcher mini-agent and its scoring function.

Invariant: no real API calls anywhere in this file.
  - AnthropicClient.create() mocked via side_effect to simulate turn sequences
  - GeminiEmbedder.embed_query() returns a fixed unit vector
  - BraveClient never called directly (all tool dispatch triggered by mock Haiku)
  - sqlite3.Connection mocked to return controlled distance values

Cosine conversion formula (unit-norm vectors):  cosine = 1 - d² / 2
  d=0.5  → cosine=0.875  → keep
  d=1.0  → cosine=0.500  → weak
  d=1.2  → cosine=0.280  → drop
  d=1.5  → cosine=0.000  → drop (clamped from negative)
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.researcher import (
    MAX_ACTIONS,
    MAX_COST_USD,
    _exec_score_relevance,
    run_researcher,
)
from app.models import Audience, ContentBrief, FunnelStage, ToneRegister


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tool_resp(name: str, tool_input: dict, tool_id: str = "toolu_001", cost: float = 0.001):
    """Build a mock Anthropic response that requests one tool call."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = tool_input

    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = [block]
    return resp, cost


def _end_resp(cost: float = 0.0005):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = []
    return resp, cost


def _make_brief() -> ContentBrief:
    return ContentBrief(
        goal="Explain how bioenergetic wellness supports cellular energy levels.",
        audience=Audience.CONSUMER,
        funnel_stage=FunnelStage.COLD,
        tone=ToneRegister.CONVERSATIONAL,
        topic_focus="chronic fatigue",
    )


def _make_clients(db_distance: float = 0.5):
    """Return (anthropic, gemini, brave, db_conn) with sensible mock defaults."""
    anthropic = MagicMock()
    gemini = MagicMock()
    gemini.embed_query.return_value = [0.1] * 3072
    brave = MagicMock()
    brave.search.return_value = []
    brave.read_url.return_value = ""
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"distance": db_distance}
    return anthropic, gemini, brave, conn


# ─── _exec_score_relevance unit tests ─────────────────────────────────────────

class TestExecScoreRelevance:
    """Deterministic scoring function — unit tested independently of the loop."""

    def _conn(self, distance: float):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = {"distance": distance}
        return conn

    def _gemini(self):
        g = MagicMock()
        g.embed_query.return_value = [0.1] * 3072
        return g

    def test_high_distance_returns_keep(self):
        # d=0.5 → cosine = 1 - 0.25/2 = 0.875 → keep
        result = _exec_score_relevance(
            {"text": "bioenergetics cellular health", "title": "T", "url": "u"},
            self._gemini(), self._conn(0.5),
        )
        assert result["label"] == "keep"
        assert result["score"] == pytest.approx(0.875, abs=0.001)

    def test_mid_distance_returns_weak(self):
        # d=1.0 → cosine = 1 - 1.0/2 = 0.500 → weak
        result = _exec_score_relevance(
            {"text": "general wellness trends", "title": "T", "url": "u"},
            self._gemini(), self._conn(1.0),
        )
        assert result["label"] == "weak"
        assert result["score"] == pytest.approx(0.500, abs=0.001)

    def test_high_distance_returns_drop(self):
        # d=1.2 → cosine = 1 - 1.44/2 = 0.280 → drop
        result = _exec_score_relevance(
            {"text": "celebrity diet weight loss", "title": "T", "url": "u"},
            self._gemini(), self._conn(1.2),
        )
        assert result["label"] == "drop"
        assert result["score"] < 0.40

    def test_very_high_distance_clamped_to_zero(self):
        # d=1.5 → cosine = 1 - 2.25/2 = -0.125 → clamped to 0.0
        result = _exec_score_relevance(
            {"text": "completely unrelated text", "title": "T", "url": "u"},
            self._gemini(), self._conn(1.5),
        )
        assert result["label"] == "drop"
        assert result["score"] == 0.0

    def test_empty_corpus_returns_drop_with_error(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        result = _exec_score_relevance(
            {"text": "any text", "title": "T", "url": "u"},
            self._gemini(), conn,
        )
        assert result["label"] == "drop"
        assert result.get("error") == "empty_corpus"

    def test_gemini_failure_returns_drop_with_error(self):
        gemini = MagicMock()
        gemini.embed_query.side_effect = RuntimeError("Gemini quota exceeded")
        result = _exec_score_relevance(
            {"text": "any text", "title": "T", "url": "u"},
            gemini, MagicMock(),
        )
        assert result["label"] == "drop"
        assert result.get("error") == "embed_failed"

    def test_text_truncated_to_gemini_limit(self):
        gemini = MagicMock()
        gemini.embed_query.return_value = [0.1] * 3072
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = {"distance": 0.5}

        long_text = "x" * 10_000
        _exec_score_relevance(
            {"text": long_text, "title": "T", "url": "u"},
            gemini, conn,
        )
        # embed_query receives text ≤ 4000 chars
        called_text = gemini.embed_query.call_args[0][0]
        assert len(called_text) <= 4000


# ─── run_researcher loop tests ────────────────────────────────────────────────

@patch("agents.researcher._load_system_prompt", return_value="mock system prompt")
class TestRunResearcher:

    def test_brave_empty_returns_no_context(self, _mock_prompt):
        anthropic, gemini, brave, conn = _make_clients()
        anthropic.create.side_effect = [
            _tool_resp("search_web", {"query": "fatigue bioenergetics"}),
            _end_resp(),
        ]
        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-001")

        assert result.findings == []
        assert result.no_context_reason == "no trend context available"

    def test_keep_score_adds_finding(self, _mock_prompt):
        # d=0.5 → cosine=0.875 → keep
        anthropic, gemini, brave, conn = _make_clients(db_distance=0.5)
        anthropic.create.side_effect = [
            _tool_resp("score_relevance_to_corpus", {
                "text": "Bioenergetic field supports ATP at cellular level.",
                "title": "Bioenergetics Study",
                "url": "https://example.com/study",
            }),
            _end_resp(),
        ]
        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-002")

        assert len(result.findings) == 1
        assert result.findings[0].relevance_label == "keep"
        assert result.findings[0].url == "https://example.com/study"
        assert result.no_context_reason is None

    def test_weak_score_adds_finding(self, _mock_prompt):
        # d=1.0 → cosine=0.500 → weak
        anthropic, gemini, brave, conn = _make_clients(db_distance=1.0)
        anthropic.create.side_effect = [
            _tool_resp("score_relevance_to_corpus", {
                "text": "Wellness trends show interest in energy management.",
                "title": "Wellness Trends",
                "url": "https://example.com/wellness",
            }),
            _end_resp(),
        ]
        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-003")

        assert len(result.findings) == 1
        assert result.findings[0].relevance_label == "weak"

    def test_drop_score_excluded_from_findings(self, _mock_prompt):
        # d=1.5 → drop
        anthropic, gemini, brave, conn = _make_clients(db_distance=1.5)
        anthropic.create.side_effect = [
            _tool_resp("score_relevance_to_corpus", {
                "text": "Celebrity diet trends 2025.",
                "title": "Diet News",
                "url": "https://example.com/diet",
            }),
            _end_resp(),
        ]
        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-004")

        assert result.findings == []

    def test_two_consecutive_drops_inject_hint(self, _mock_prompt):
        # d=1.5 → drop for all score calls
        anthropic, gemini, brave, conn = _make_clients(db_distance=1.5)

        captured: list[dict] = []

        def capture_create(**kwargs):
            # Inspect tool_result messages for hints injected by Python
            for msg in kwargs.get("messages", []):
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            try:
                                captured.append(json.loads(item["content"]))
                            except Exception:
                                pass
            return _end_resp()

        anthropic.create.side_effect = [
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 1", "title": "T1", "url": "u1"}, tool_id="t1"),
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 2", "title": "T2", "url": "u2"}, tool_id="t2"),
            # After 2 drops, hint injected; Haiku ends turn
            MagicMock(side_effect=capture_create),
        ]
        # Use side_effect list for first two, then capture_create
        anthropic.create.side_effect = [
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 1", "title": "T1", "url": "t1"}, tool_id="t1"),
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 2", "title": "T2", "url": "t2"}, tool_id="t2"),
            _end_resp(),
        ]
        # Wrap create to capture what messages it receives
        original_side_effect = list(anthropic.create.side_effect)
        call_index = [0]

        def side_effect_with_capture(*args, **kwargs):
            for msg in kwargs.get("messages", []):
                if msg["role"] == "user" and isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            try:
                                captured.append(json.loads(item["content"]))
                            except Exception:
                                pass
            val = original_side_effect[call_index[0]]
            call_index[0] += 1
            return val

        anthropic.create.side_effect = side_effect_with_capture

        run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-005")

        hints = [r.get("hint", "") for r in captured if r.get("hint")]
        assert len(hints) >= 1, "Expected a reformulation hint after 2 consecutive drops"
        assert "reformulate" in hints[0].lower() or "different" in hints[0].lower() or "abandon" in hints[0].lower()

    def test_reformulate_then_two_more_drops_returns_no_context(self, _mock_prompt):
        # All score calls drop → 2 drops → hint → 2 more drops → early return
        anthropic, gemini, brave, conn = _make_clients(db_distance=1.5)
        anthropic.create.side_effect = [
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 1", "title": "T1", "url": "u1"}, tool_id="t1"),
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 2", "title": "T2", "url": "u2"}, tool_id="t2"),
            # After hint (reformulated=True), 2 more drops → early return
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 3", "title": "T3", "url": "u3"}, tool_id="t3"),
            _tool_resp("score_relevance_to_corpus",
                {"text": "irrelevant 4", "title": "T4", "url": "u4"}, tool_id="t4"),
        ]

        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-006")

        assert result.findings == []
        assert result.no_context_reason == "no trend context available"
        assert result.actions_used == 4

    def test_action_cap_terminates_loop(self, _mock_prompt):
        # All high scores (keeps), but hit the 5-action cap
        anthropic, gemini, brave, conn = _make_clients(db_distance=0.5)
        anthropic.create.side_effect = [
            _tool_resp(
                "score_relevance_to_corpus",
                {"text": f"content {i}", "title": f"T{i}", "url": f"https://ex.com/{i}"},
                tool_id=f"t{i}",
            )
            for i in range(MAX_ACTIONS)
        ]

        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-007")

        assert result.actions_used == MAX_ACTIONS
        assert len(result.findings) == MAX_ACTIONS

    def test_cost_cap_terminates_loop(self, _mock_prompt):
        # Each turn costs $0.06; after 2 turns cost_usd ≥ $0.10 → loop exits
        anthropic, gemini, brave, conn = _make_clients(db_distance=0.5)
        anthropic.create.side_effect = [
            _tool_resp(
                "score_relevance_to_corpus",
                {"text": f"content {i}", "title": f"T{i}", "url": f"https://ex.com/{i}"},
                tool_id=f"t{i}",
                cost=0.06,
            )
            for i in range(5)
        ]

        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-008")

        assert result.cost_usd >= MAX_COST_USD
        assert result.actions_used <= 2

    def test_snippet_truncated_in_finding(self, _mock_prompt):
        anthropic, gemini, brave, conn = _make_clients(db_distance=0.5)
        long_text = "bioenergetic " * 200  # > 600 chars
        anthropic.create.side_effect = [
            _tool_resp("score_relevance_to_corpus",
                {"text": long_text, "title": "T", "url": "https://ex.com/t"}),
            _end_resp(),
        ]

        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-009")

        assert len(result.findings) == 1
        assert len(result.findings[0].snippet) <= 600

    def test_end_turn_on_first_response_returns_no_context(self, _mock_prompt):
        # Haiku immediately ends turn without calling any tool
        anthropic, gemini, brave, conn = _make_clients()
        anthropic.create.side_effect = [_end_resp()]

        result = run_researcher(_make_brief(), anthropic, gemini, brave, conn, "t-010")

        assert result.findings == []
        assert result.no_context_reason == "no trend context available"
        assert result.actions_used == 0
