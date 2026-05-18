"""
Tests for S7 endpoints: GET /runs/{trace_id}/detail and GET /runs/{trace_id}/replay.

All tests are deterministic — no LLM calls, no API keys required.
Tests mock app.state.obs_conn and app.state.anthropic so the
FastAPI test client can exercise the endpoint logic without a live server.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_obs_conn() -> sqlite3.Connection:
    """In-memory obs.db with S7 schema. check_same_thread=False because
    the TestClient dispatches requests in a separate thread from the fixture."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE runs (
            trace_id        TEXT PRIMARY KEY,
            brief_json      TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'running',
            start_ts        TEXT NOT NULL,
            end_ts          TEXT,
            total_cost_usd  REAL NOT NULL DEFAULT 0.0,
            turns_used      INTEGER NOT NULL DEFAULT 0,
            is_replay       INTEGER NOT NULL DEFAULT 0,
            replayed_from   TEXT
        );
        CREATE TABLE agent_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id    TEXT NOT NULL,
            agent_name  TEXT NOT NULL,
            model       TEXT NOT NULL,
            tokens_in   INTEGER NOT NULL DEFAULT 0,
            tokens_out  INTEGER NOT NULL DEFAULT 0,
            cost_usd    REAL NOT NULL DEFAULT 0.0,
            latency_ms  INTEGER NOT NULL DEFAULT 0,
            tool_calls  INTEGER NOT NULL DEFAULT 0,
            error       TEXT,
            ts          TEXT NOT NULL
        );
        CREATE TABLE tool_call_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id    TEXT NOT NULL,
            agent_name  TEXT NOT NULL,
            tool_name   TEXT NOT NULL,
            input_json  TEXT NOT NULL,
            output_json TEXT NOT NULL,
            ts          TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


_BRIEF_JSON = json.dumps({
    "goal": "Introduce miHealth to consumers experiencing chronic fatigue",
    "audience": "consumer",
    "funnel_stage": "cold",
    "platforms": ["linkedin", "email"],
    "topic_focus": None,
    "product_focus": [],
    "tone": "conversational",
    "format_intent": None,
})


def _seed_run(
    conn: sqlite3.Connection,
    trace_id: str = "abc-123",
    is_replay: bool = False,
    replayed_from: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
        (trace_id, _BRIEF_JSON, "complete", "2026-05-16T10:00:00+00:00",
         "2026-05-16T10:01:00+00:00", 0.0167, 6, int(is_replay), replayed_from),
    )
    conn.execute(
        "INSERT INTO agent_calls (trace_id,agent_name,model,tokens_in,tokens_out,cost_usd,latency_ms,tool_calls,ts) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (trace_id, "orchestrator", "claude-haiku-4-5-20251001", 800, 120, 0.0004, 950, 3, "2026-05-16T10:00:01+00:00"),
    )
    conn.execute(
        "INSERT INTO tool_call_events (trace_id,agent_name,tool_name,input_json,output_json,ts) "
        "VALUES (?,?,?,?,?,?)",
        (trace_id, "orchestrator", "call_writer",
         json.dumps({"platform": "linkedin"}),
         json.dumps({"status": "ok", "platform": "linkedin"}),
         "2026-05-16T10:00:02+00:00"),
    )
    conn.commit()


@pytest.fixture
def client():
    """TestClient with mocked app.state — no real DB, no API clients."""
    from app.main import app

    obs_conn = _make_obs_conn()
    _seed_run(obs_conn, "abc-123")
    _seed_run(obs_conn, "replay-456", is_replay=True, replayed_from="abc-123")

    app.state.obs_conn = obs_conn
    app.state.anthropic = MagicMock()
    app.state.gemini = MagicMock()
    app.state.brave = MagicMock()
    app.state.corpus_conn = MagicMock()
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /runs/{trace_id}/detail
# ---------------------------------------------------------------------------

class TestRunDetail:
    def test_returns_run_agent_calls_tool_events(self, client):
        r = client.get("/runs/abc-123/detail")
        assert r.status_code == 200
        data = r.json()
        assert data["run"]["trace_id"] == "abc-123"
        assert len(data["agent_calls"]) == 1
        assert data["agent_calls"][0]["agent_name"] == "orchestrator"
        assert len(data["tool_events"]) == 1

    def test_tool_events_include_input_summary(self, client):
        r = client.get("/runs/abc-123/detail")
        ev = r.json()["tool_events"][0]
        assert "input_summary" in ev
        assert "linkedin" in ev["input_summary"]

    def test_404_for_unknown_trace(self, client):
        r = client.get("/runs/does-not-exist/detail")
        assert r.status_code == 404

    def test_replay_run_has_replay_fields(self, client):
        r = client.get("/runs/replay-456/detail")
        assert r.status_code == 200
        run = r.json()["run"]
        assert run["is_replay"] == 1
        assert run["replayed_from"] == "abc-123"


# ---------------------------------------------------------------------------
# GET /runs/{trace_id}/replay
# ---------------------------------------------------------------------------

class TestReplayRun:
    def test_returns_new_trace_id_and_replayed_from(self, client):
        with patch("app.main.asyncio.create_task"):
            with patch("app.main._event_manager") as em:
                r = client.get("/runs/abc-123/replay")
        assert r.status_code == 200
        data = r.json()
        assert "trace_id" in data
        assert data["replayed_from"] == "abc-123"
        assert data["trace_id"] != "abc-123"

    def test_new_trace_id_is_uuid_format(self, client):
        with patch("app.main.asyncio.create_task"):
            with patch("app.main._event_manager"):
                r = client.get("/runs/abc-123/replay")
        import re
        assert re.match(r"[0-9a-f-]{36}", r.json()["trace_id"])

    def test_404_for_unknown_trace(self, client):
        r = client.get("/runs/no-such-run/replay")
        assert r.status_code == 404

    def test_422_if_brief_json_corrupt(self, client):
        obs = client.app.state.obs_conn
        obs.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
            ("bad-brief", "not-valid-json", "complete", "2026-05-16T10:00:00+00:00",
             None, 0.0, 0, 0, None),
        )
        obs.commit()
        r = client.get("/runs/bad-brief/replay")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /runs — includes is_replay and replayed_from
# ---------------------------------------------------------------------------

class TestRunsListReplayFields:
    def test_runs_list_includes_replay_fields(self, client):
        r = client.get("/runs")
        assert r.status_code == 200
        runs = r.json()
        replay_run = next((x for x in runs if x["trace_id"] == "replay-456"), None)
        assert replay_run is not None
        assert replay_run["is_replay"] is True
        assert replay_run["replayed_from"] == "abc-123"

    def test_normal_run_is_replay_false(self, client):
        r = client.get("/runs")
        normal = next((x for x in r.json() if x["trace_id"] == "abc-123"), None)
        assert normal["is_replay"] is False
        assert normal["replayed_from"] is None
