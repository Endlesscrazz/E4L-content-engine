"""
Observability sqlite store — trace_id-indexed run history.

Three tables in corpus/obs.db (separate from corpus.db — different data lifecycle):
  runs             — one row per pipeline run; status updated at completion
  agent_calls      — one row per agent invocation; written after each call returns
  tool_call_events — one row per tool_use inside any agent loop; written before returning

All writes are fire-and-forget: failures are logged but never raise. Run history
is an audit trail — a write failure must not abort a content generation run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.models import AgentTelemetry, RunRecord, ToolCallEvent

logger = logging.getLogger(__name__)

DEFAULT_OBS_DB = Path("corpus/obs.db")


# ─── Schema ───────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS runs (
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

CREATE TABLE IF NOT EXISTS agent_calls (
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

CREATE TABLE IF NOT EXISTS tool_call_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL,
    agent_name  TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    input_json  TEXT NOT NULL,
    output_json TEXT NOT NULL,
    ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_calls_trace ON agent_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_tool_events_trace ON tool_call_events(trace_id);
"""


def init_obs_db(db_path: Path = DEFAULT_OBS_DB) -> sqlite3.Connection:
    """Create obs.db and return an open connection. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # WAL mode leaves two sidecar files (obs.db-wal, obs.db-shm). If the main db
    # is deleted without removing them (e.g. manual "clear history"), SQLite raises
    # a disk I/O error on the next open. Remove orphaned sidecars preemptively.
    if not db_path.exists():
        for suf in ("-wal", "-shm"):
            (db_path.parent / (db_path.name + suf)).unlink(missing_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL mode: readers never block writers and vice versa — necessary for the
    # asyncio.to_thread write path sharing a connection with the async read path.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DDL)
    conn.commit()
    # Migrate: add S7 replay columns if obs.db predates this schema.
    # ALTER TABLE IF NOT EXISTS is not supported in SQLite 3.x, so we catch OperationalError.
    for col, defn in [
        ("is_replay", "INTEGER NOT NULL DEFAULT 0"),
        ("replayed_from", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Write helpers (fire-and-forget; never raise) ─────────────────────────────

def write_run_start(conn: sqlite3.Connection, record: RunRecord) -> None:
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO runs
              (trace_id, brief_json, status, start_ts, is_replay, replayed_from)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record.trace_id,
                record.brief_json,
                record.status,
                record.start_ts,
                int(record.is_replay),
                record.replayed_from,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("obs write_run_start failed: %s", exc)


def write_run_end(
    conn: sqlite3.Connection,
    trace_id: str,
    status: str,
    total_cost_usd: float,
    turns_used: int,
) -> None:
    try:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, end_ts = ?, total_cost_usd = ?, turns_used = ?
            WHERE trace_id = ?
            """,
            (status, _now_iso(), total_cost_usd, turns_used, trace_id),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("obs write_run_end failed: %s", exc)


def write_agent_call(conn: sqlite3.Connection, telemetry: AgentTelemetry) -> None:
    try:
        conn.execute(
            """
            INSERT INTO agent_calls
              (trace_id, agent_name, model, tokens_in, tokens_out,
               cost_usd, latency_ms, tool_calls, error, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                telemetry.trace_id,
                telemetry.agent_name,
                telemetry.model,
                telemetry.tokens_in,
                telemetry.tokens_out,
                telemetry.cost_usd,
                telemetry.latency_ms,
                telemetry.tool_calls,
                telemetry.error,
                _now_iso(),
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("obs write_agent_call failed: %s", exc)


def write_tool_event(conn: sqlite3.Connection, event: ToolCallEvent) -> None:
    try:
        conn.execute(
            """
            INSERT INTO tool_call_events
              (trace_id, agent_name, tool_name, input_json, output_json, ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.trace_id,
                event.agent_name,
                event.tool_name,
                event.input_json,
                event.output_json,
                event.ts,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("obs write_tool_event failed: %s", exc)


def query_runs(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    """Return recent runs as dicts — used by the S6 /runs endpoint."""
    # Commit clears any implicit transaction so the SELECT sees the latest writes.
    conn.commit()
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY start_ts DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def _summarize_input(input_json: str, max_len: int = 80) -> str:
    """Extract a readable one-line summary from a tool input JSON blob."""
    try:
        data = json.loads(input_json)
        parts = [f"{k}={str(v)[:40]}" for k, v in list(data.items())[:2] if v]
        return ("  ".join(parts))[:max_len]
    except Exception:
        return input_json[:max_len]


def delete_run(conn: sqlite3.Connection, trace_id: str) -> bool:
    """Delete a run and all its telemetry rows. Returns True if the run existed."""
    conn.commit()
    exists = conn.execute("SELECT 1 FROM runs WHERE trace_id=?", (trace_id,)).fetchone()
    if not exists:
        return False
    conn.execute("DELETE FROM tool_call_events WHERE trace_id=?", (trace_id,))
    conn.execute("DELETE FROM agent_calls WHERE trace_id=?", (trace_id,))
    conn.execute("DELETE FROM runs WHERE trace_id=?", (trace_id,))
    conn.commit()
    return True


def query_run_detail(conn: sqlite3.Connection, trace_id: str) -> dict:
    """Return full per-agent trace for a run — used by S7 detail + replay endpoints."""
    # Commit clears any implicit transaction so the SELECT sees the latest writes.
    conn.commit()
    run_row = conn.execute(
        "SELECT * FROM runs WHERE trace_id = ?", (trace_id,)
    ).fetchone()
    if not run_row:
        return {}

    agent_rows = conn.execute(
        "SELECT * FROM agent_calls WHERE trace_id = ? ORDER BY ts", (trace_id,)
    ).fetchall()

    tool_rows = conn.execute(
        "SELECT * FROM tool_call_events WHERE trace_id = ? ORDER BY ts", (trace_id,)
    ).fetchall()

    return {
        "run": dict(run_row),
        "agent_calls": [dict(r) for r in agent_rows],
        "tool_events": [
            {**dict(r), "input_summary": _summarize_input(r["input_json"])}
            for r in tool_rows
        ],
    }
