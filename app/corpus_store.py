"""
sqlite-vec corpus store.

Schema:
  chunks          — metadata + content (TEXT PRIMARY KEY chunk_id)
  vec_embeddings  — vec0 virtual table (TEXT PRIMARY KEY chunk_id, float[3072])

The vec0 TEXT primary key is supported in sqlite-vec 0.1.6 and later.
Similarity search uses the vec0 MATCH operator; the outer query joins back
to chunks for metadata. never_in_generated_content chunks are excluded from
retrieval at query time (not at insert time — we keep them for audit).

Audience and product filters are applied post-join because vec0 does not
support WHERE predicates inside the MATCH subquery.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from app.models import SourceChunk

DEFAULT_DB = Path("corpus/corpus.db")


def get_conn(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id            TEXT PRIMARY KEY,
            doc_name            TEXT NOT NULL,
            doc_type            TEXT NOT NULL,
            content             TEXT NOT NULL,
            audience_tags       TEXT DEFAULT '[]',
            product_associations TEXT DEFAULT '[]',
            is_voice_anchor     INTEGER DEFAULT 0,
            do_not_discuss      INTEGER DEFAULT 0,
            do_not_discuss_mode TEXT,
            corpus_conflict     INTEGER DEFAULT 0,
            annotation_source   TEXT DEFAULT 'inferred',
            created_at          TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
            chunk_id  TEXT PRIMARY KEY,
            embedding float[3072]
        );
    """)
    conn.commit()
    return conn


def insert_chunk(
    conn: sqlite3.Connection,
    chunk: SourceChunk,
    embedding: list[float],
) -> None:
    annotation_source = (
        "human_reviewed" if (chunk.do_not_discuss or chunk.is_voice_anchor) else "inferred"
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO chunks
          (chunk_id, doc_name, doc_type, content, audience_tags,
           product_associations, is_voice_anchor, do_not_discuss,
           do_not_discuss_mode, corpus_conflict, annotation_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk.chunk_id,
            chunk.doc_name,
            chunk.doc_type,
            chunk.content,
            json.dumps(chunk.audience_tags),
            json.dumps(chunk.product_associations),
            int(chunk.is_voice_anchor),
            int(chunk.do_not_discuss),
            chunk.do_not_discuss_mode,
            int(chunk.corpus_conflict),
            annotation_source,
        ),
    )
    conn.execute(
        "INSERT OR REPLACE INTO vec_embeddings (chunk_id, embedding) VALUES (?, ?)",
        (chunk.chunk_id, json.dumps(embedding)),
    )


def query_similar(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int = 10,
    exclude_do_not_discuss: bool = True,
    audience_filter: str | None = None,
    product_filter: str | None = None,
) -> list[SourceChunk]:
    """
    Return top-k chunks by cosine similarity.

    Fetches top_k * 4 from vec0 to have headroom after filtering.
    never_in_generated_content chunks are always excluded when
    exclude_do_not_discuss=True (do_not_volunteer chunks stay in).
    """
    candidate_limit = top_k * 4
    rows: list[Any] = conn.execute(
        """
        SELECT c.*, ve.distance
        FROM chunks c
        JOIN (
            SELECT chunk_id, distance
            FROM vec_embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
        ) ve ON c.chunk_id = ve.chunk_id
        WHERE (
            ? = 0
            OR c.do_not_discuss_mode IS NULL
            OR c.do_not_discuss_mode != 'never_in_generated_content'
        )
        ORDER BY ve.distance
        """,
        (json.dumps(query_embedding), candidate_limit, int(exclude_do_not_discuss)),
    ).fetchall()

    results: list[SourceChunk] = []
    for row in rows:
        if audience_filter:
            tags = json.loads(row["audience_tags"] or "[]")
            if audience_filter not in tags:
                continue
        if product_filter:
            prods = json.loads(row["product_associations"] or "[]")
            if product_filter not in prods:
                continue
        results.append(_row_to_chunk(row))
        if len(results) >= top_k:
            break

    return results


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> SourceChunk | None:
    row = conn.execute(
        "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    return _row_to_chunk(row) if row else None


def get_voice_anchors(conn: sqlite3.Connection) -> list[SourceChunk]:
    rows = conn.execute(
        "SELECT * FROM chunks WHERE is_voice_anchor = 1"
    ).fetchall()
    return [_row_to_chunk(r) for r in rows]


def _row_to_chunk(row: sqlite3.Row) -> SourceChunk:
    return SourceChunk(
        chunk_id=row["chunk_id"],
        doc_name=row["doc_name"],
        doc_type=row["doc_type"],
        content=row["content"],
        audience_tags=json.loads(row["audience_tags"] or "[]"),
        product_associations=json.loads(row["product_associations"] or "[]"),
        is_voice_anchor=bool(row["is_voice_anchor"]),
        do_not_discuss=bool(row["do_not_discuss"]),
        do_not_discuss_mode=row["do_not_discuss_mode"],
        corpus_conflict=bool(row["corpus_conflict"]),
    )
