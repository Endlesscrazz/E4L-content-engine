"""
S1 tests: chunkers, conflict detection, corpus store.
All deterministic — no LLM calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.chunkers import (
    chunk_ai_version,
    chunk_narrative,
    chunk_product_doc,
    chunk_research_summary,
)
from app.corpus_store import get_chunk, init_db, insert_chunk, query_similar
from app.models import SourceChunk
from scripts.ingest_corpus import detect_conflicts

SOURCE = Path("source_docs")


# ── chunk_narrative ────────────────────────────────────────────────────────────

class TestChunkNarrative:
    def test_origin_story_chunk_count(self):
        chunks = chunk_narrative(SOURCE / "Energy4Life Origin Story.docx")
        assert len(chunks) >= 5

    def test_differentiation_chunk_count(self):
        chunks = chunk_narrative(SOURCE / "Energy4Life Differentiation.docx")
        assert len(chunks) >= 5

    def test_unique_chunk_ids_origin(self):
        chunks = chunk_narrative(SOURCE / "Energy4Life Origin Story.docx")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_empty_chunks(self):
        chunks = chunk_narrative(SOURCE / "Energy4Life Origin Story.docx")
        assert all(c.content.strip() for c in chunks)

    def test_peter_fraser_chunk_present(self):
        """The editorial 'NOT to be discussed' note must survive as its own chunk."""
        chunks = chunk_narrative(SOURCE / "Energy4Life Origin Story.docx")
        flagged = [c for c in chunks if "NOT to be discussed" in c.content]
        assert len(flagged) == 1

    def test_doc_type(self):
        chunks = chunk_narrative(SOURCE / "Energy4Life Origin Story.docx")
        assert all(c.doc_type == "narrative" for c in chunks)


# ── chunk_ai_version ──────────────────────────────────────────────────────────

class TestChunkAiVersion:
    def test_section_count(self):
        chunks = chunk_ai_version(SOURCE / "AI Version of Restore Your Energy with Bioenergetics.docx")
        assert len(chunks) >= 10

    def test_unique_chunk_ids(self):
        chunks = chunk_ai_version(SOURCE / "AI Version of Restore Your Energy with Bioenergetics.docx")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_empty_chunks(self):
        chunks = chunk_ai_version(SOURCE / "AI Version of Restore Your Energy with Bioenergetics.docx")
        assert all(c.content.strip() for c in chunks)

    def test_streeter_chunk_present(self):
        """AI Version cites the Streeter study (with 200 — the conflicting number)."""
        chunks = chunk_ai_version(SOURCE / "AI Version of Restore Your Energy with Bioenergetics.docx")
        matches = [c for c in chunks if "Streeter" in c.content]
        assert len(matches) >= 1

    def test_doc_type(self):
        chunks = chunk_ai_version(SOURCE / "AI Version of Restore Your Energy with Bioenergetics.docx")
        assert all(c.doc_type == "ai_version" for c in chunks)


# ── chunk_product_doc ─────────────────────────────────────────────────────────

class TestChunkProductDoc:
    def test_mihealth_chunk_count(self):
        chunks = chunk_product_doc(SOURCE / "E4L miHealth Product Summary w_o Original.docx")
        assert len(chunks) >= 3

    def test_bws_chunk_count(self):
        chunks = chunk_product_doc(SOURCE / "Bioenergetic Wellness System (BWS) Product Summary.docx")
        assert len(chunks) >= 2

    def test_unique_ids_mihealth(self):
        chunks = chunk_product_doc(SOURCE / "E4L miHealth Product Summary w_o Original.docx")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_empty_chunks(self):
        chunks = chunk_product_doc(SOURCE / "E4L miHealth Product Summary w_o Original.docx")
        assert all(c.content.strip() for c in chunks)

    def test_doc_type(self):
        chunks = chunk_product_doc(SOURCE / "E4L miHealth Product Summary w_o Original.docx")
        assert all(c.doc_type == "product_doc" for c in chunks)

    def test_bws_tws_chunk_separated(self):
        """The TWS legacy note must be a standalone chunk (after the *** separator)."""
        chunks = chunk_product_doc(SOURCE / "Bioenergetic Wellness System (BWS) Product Summary.docx")
        tws_chunks = [c for c in chunks if "Total Wellness System" in c.content or "has not been sold since 2017" in c.content]
        assert len(tws_chunks) >= 1


# ── chunk_research_summary ────────────────────────────────────────────────────

class TestChunkResearchSummary:
    def test_finding_count(self):
        chunks = chunk_research_summary(SOURCE / "Energy4Life Research Summary.txt")
        assert len(chunks) >= 8

    def test_streeter_finding_present(self):
        """Research Summary must have the Streeter study as a finding (240 participants)."""
        chunks = chunk_research_summary(SOURCE / "Energy4Life Research Summary.txt")
        streeter = [c for c in chunks if "Streeter" in c.content]
        assert len(streeter) >= 1

    def test_no_empty_chunks(self):
        chunks = chunk_research_summary(SOURCE / "Energy4Life Research Summary.txt")
        assert all(c.content.strip() for c in chunks)

    def test_doc_type(self):
        chunks = chunk_research_summary(SOURCE / "Energy4Life Research Summary.txt")
        assert all(c.doc_type == "research_finding" for c in chunks)


# ── Conflict detection ────────────────────────────────────────────────────────

class TestConflictDetection:
    def test_streeter_conflict_flagged(self):
        """Both the 200-participant (AI Version) and 240-participant (Research Summary)
        Streeter chunks must be flagged corpus_conflict=True."""
        ai_chunks = chunk_ai_version(
            SOURCE / "AI Version of Restore Your Energy with Bioenergetics.docx"
        )
        rs_chunks = chunk_research_summary(SOURCE / "Energy4Life Research Summary.txt")
        all_chunks = detect_conflicts(ai_chunks + rs_chunks)

        conflict_chunks = [c for c in all_chunks if c.corpus_conflict]
        assert len(conflict_chunks) >= 2

        combined_text = " ".join(c.content for c in conflict_chunks)
        assert "200" in combined_text
        assert "240" in combined_text

    def test_non_conflicting_chunks_not_flagged(self):
        """Narrative docs with no conflicting numeric claims must not be flagged."""
        chunks = chunk_narrative(SOURCE / "Energy4Life Origin Story.docx")
        flagged = detect_conflicts(chunks)
        assert not any(c.corpus_conflict for c in flagged)


# ── Corpus store ──────────────────────────────────────────────────────────────

def _fake_embedding(seed: float = 1.0) -> list[float]:
    return [seed] + [0.0] * 3071


@pytest.fixture
def mem_db(tmp_path):
    return init_db(tmp_path / "test.db")


class TestCorpusStore:
    def test_insert_and_retrieve(self, mem_db):
        chunk = SourceChunk(
            chunk_id="test_0001",
            doc_name="test.docx",
            doc_type="narrative",
            content="Test content about energy and wellness.",
        )
        insert_chunk(mem_db, chunk, _fake_embedding())
        mem_db.commit()
        result = get_chunk(mem_db, "test_0001")
        assert result is not None
        assert result.chunk_id == "test_0001"
        assert result.content == chunk.content

    def test_do_not_discuss_round_trips(self, mem_db):
        chunk = SourceChunk(
            chunk_id="dnd_0001",
            doc_name="origin.docx",
            doc_type="narrative",
            content="For AI Only: this is NOT to be discussed.",
            do_not_discuss=True,
            do_not_discuss_mode="never_in_generated_content",
        )
        insert_chunk(mem_db, chunk, _fake_embedding(0.5))
        mem_db.commit()
        result = get_chunk(mem_db, "dnd_0001")
        assert result.do_not_discuss is True
        assert result.do_not_discuss_mode == "never_in_generated_content"

    def test_corpus_conflict_round_trips(self, mem_db):
        chunk = SourceChunk(
            chunk_id="conflict_0001",
            doc_name="research.txt",
            doc_type="research_finding",
            content="Streeter study: 200 participants.",
            corpus_conflict=True,
        )
        insert_chunk(mem_db, chunk, _fake_embedding())
        mem_db.commit()
        result = get_chunk(mem_db, "conflict_0001")
        assert result.corpus_conflict is True

    def test_query_excludes_never_in_generated_content(self, mem_db):
        safe = SourceChunk(
            chunk_id="safe_0001",
            doc_name="d.docx",
            doc_type="narrative",
            content="Safe content about bioenergetics.",
        )
        blocked = SourceChunk(
            chunk_id="blocked_0001",
            doc_name="d.docx",
            doc_type="narrative",
            content="Peter Fraser death details.",
            do_not_discuss=True,
            do_not_discuss_mode="never_in_generated_content",
        )
        insert_chunk(mem_db, safe, _fake_embedding(1.0))
        insert_chunk(mem_db, blocked, _fake_embedding(1.0))
        mem_db.commit()

        results = query_similar(
            mem_db, _fake_embedding(1.0), top_k=10, exclude_do_not_discuss=True
        )
        ids = {r.chunk_id for r in results}
        assert "safe_0001" in ids
        assert "blocked_0001" not in ids

    def test_query_includes_do_not_volunteer_chunks(self, mem_db):
        """do_not_volunteer chunks ARE retrievable — Validator decides whether to ship."""
        chunk = SourceChunk(
            chunk_id="dnv_0001",
            doc_name="bws.docx",
            doc_type="product_doc",
            content="TWS has not been sold since 2017.",
            do_not_discuss=True,
            do_not_discuss_mode="do_not_volunteer",
        )
        insert_chunk(mem_db, chunk, _fake_embedding(1.0))
        mem_db.commit()

        results = query_similar(
            mem_db, _fake_embedding(1.0), top_k=10, exclude_do_not_discuss=True
        )
        ids = {r.chunk_id for r in results}
        assert "dnv_0001" in ids
