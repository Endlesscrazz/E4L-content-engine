#!/usr/bin/env python3
"""
Corpus ingestion script.

Usage:
    python scripts/ingest_corpus.py source_docs/
    python scripts/ingest_corpus.py source_docs/ --dry-run   # chunk + annotate, no DB write

Flow:
  1. Route each doc to its chunker by filename
  2. Apply corpus_annotations.yaml: audience tags, voice anchors, product associations
  3. Apply do_not_discuss.yaml: flag restricted chunks
  4. Detect cross-chunk numeric conflicts (e.g. Streeter study 200 vs 240)
  5. Embed all chunks via Gemini text-embedding-004 (3072-dim)
  6. Persist to sqlite-vec (INSERT OR REPLACE — idempotent)

Conflict detection rationale: the Streeter study participant count differs between
the Research Summary (240) and the AI Version (200). Both chunks are flagged
corpus_conflict=True so the Validator and Writers know not to assert a definitive
number without hedging.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Allow running from project root or scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunkers import (
    chunk_ai_version,
    chunk_narrative,
    chunk_product_doc,
    chunk_research_summary,
)
from app.clients import GeminiEmbedder
from app.corpus_store import init_db, insert_chunk
from app.models import SourceChunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
_ANNOTATIONS_FILE = _CORPUS_DIR / "corpus_annotations.yaml"
_DND_FILE = _CORPUS_DIR / "do_not_discuss.yaml"

# Known numeric conflicts across the corpus.
# Both chunks that reference the same study with different numbers get flagged.
_CONFLICT_SIGNATURES = [
    {
        "name": "Streeter / Centre for Biofield Sciences",
        "context_re": re.compile(r"Streeter|Centre for Biofield Sciences", re.I),
        "numeric_re": re.compile(r"(\d+)\s+participants", re.I),
    },
]


# ── Routing ───────────────────────────────────────────────────────────────────

def _route(doc_path: Path) -> list[SourceChunk]:
    name = doc_path.name.lower()
    if "origin story" in name or "differentiation" in name:
        return chunk_narrative(doc_path)
    if "ai version" in name:
        return chunk_ai_version(doc_path)
    if "mihea" in name or "bws" in name or "bioenergetic wellness" in name:
        return chunk_product_doc(doc_path)
    if name.endswith(".txt"):
        return chunk_research_summary(doc_path)
    raise ValueError(f"No chunker registered for: {doc_path.name}")


# ── Annotation application ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalize smart quotes to straight for pattern matching only."""
    return (
        text.replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .lower()
    )


def _apply_annotations(
    chunks: list[SourceChunk],
    annotations: dict,
    dnd_flags: list[dict],
) -> list[SourceChunk]:
    doc_defaults: dict = annotations.get("doc_level_defaults", {})
    voice_anchors: list[dict] = annotations.get("voice_anchors", [])
    product_patterns: list[dict] = annotations.get("product_pattern_associations", [])

    for chunk in chunks:
        # 1. Doc-level audience tags
        defaults = doc_defaults.get(chunk.doc_name, {})
        if "audience_tags" in defaults:
            chunk.audience_tags = list(defaults["audience_tags"])

        # 2. Product associations via regex (inferred)
        found: list[str] = []
        for entry in product_patterns:
            if re.search(entry["pattern"], chunk.content, re.I):
                p = entry["product"]
                if p not in found:
                    found.append(p)
        chunk.product_associations = found

        # 3. Voice anchor detection (human_reviewed)
        for va in voice_anchors:
            name_match = va.get("doc_name_contains", "").lower()
            if name_match and name_match not in chunk.doc_name.lower():
                continue
            patterns = va.get("content_contains_any", [])
            norm_content = _normalize(chunk.content)
            if any(_normalize(p) in norm_content for p in patterns):
                chunk.is_voice_anchor = True
                break

        # 4. do_not_discuss chunk-level flags (human_reviewed)
        for flag in dnd_flags:
            patterns = flag.get("chunk_content_patterns") or []
            if not patterns:
                continue
            norm_content = _normalize(chunk.content)
            if any(_normalize(p) in norm_content for p in patterns):
                chunk.do_not_discuss = True
                chunk.do_not_discuss_mode = flag["mode"]
                break

    return chunks


# ── Conflict detection ────────────────────────────────────────────────────────

def detect_conflicts(chunks: list[SourceChunk]) -> list[SourceChunk]:
    """
    Flag corpus_conflict=True on any pair of chunks where the same named study
    is cited with different numeric values.
    Currently detects: Streeter study (200 vs 240 participants).
    """
    for sig in _CONFLICT_SIGNATURES:
        matches: list[tuple[SourceChunk, set[str]]] = []
        for chunk in chunks:
            if sig["context_re"].search(chunk.content):
                nums = set(sig["numeric_re"].findall(chunk.content))
                if nums:
                    matches.append((chunk, nums))

        if len(matches) < 2:
            continue

        all_nums: set[str] = set()
        for _, nums in matches:
            all_nums |= nums

        if len(all_nums) > 1:
            log.warning(
                "Corpus conflict  '%s': chunks %s cite conflicting numbers %s",
                sig["name"],
                [c.chunk_id for c, _ in matches],
                sorted(all_nums),
            )
            for chunk, _ in matches:
                chunk.corpus_conflict = True

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(".env.local", override=True)
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ingest E4L source corpus into sqlite-vec."
    )
    parser.add_argument("source_dir", type=Path, help="Directory containing source docs.")
    parser.add_argument(
        "--db",
        type=Path,
        default=_CORPUS_DIR / "corpus.db",
        help="Output sqlite-vec database path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk, annotate, and report — skip embedding and DB write.",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        log.error("source_dir %s not found", args.source_dir)
        sys.exit(1)

    annotations = (
        yaml.safe_load(_ANNOTATIONS_FILE.read_text())
        if _ANNOTATIONS_FILE.exists()
        else {}
    )
    dnd_data = (
        yaml.safe_load(_DND_FILE.read_text()) if _DND_FILE.exists() else {}
    )
    dnd_flags: list[dict] = dnd_data.get("do_not_discuss_flags", [])

    # ── Chunk all docs ────────────────────────────────────────────────────────
    all_chunks: list[SourceChunk] = []
    doc_exts = {".docx", ".txt"}
    source_files = sorted(
        f for f in args.source_dir.iterdir() if f.suffix.lower() in doc_exts
    )

    if not source_files:
        log.error("No .docx or .txt files found in %s", args.source_dir)
        sys.exit(1)

    for doc_path in source_files:
        chunks = _route(doc_path)
        log.info("%-65s  %d chunks", doc_path.name, len(chunks))
        all_chunks.extend(chunks)

    log.info("Total: %d chunks across %d docs", len(all_chunks), len(source_files))

    # ── Annotate ──────────────────────────────────────────────────────────────
    all_chunks = _apply_annotations(all_chunks, annotations, dnd_flags)
    all_chunks = detect_conflicts(all_chunks)

    dnd_count = sum(1 for c in all_chunks if c.do_not_discuss)
    conflict_count = sum(1 for c in all_chunks if c.corpus_conflict)
    voice_count = sum(1 for c in all_chunks if c.is_voice_anchor)
    log.info(
        "Annotations: %d do_not_discuss  %d corpus_conflict  %d voice_anchors",
        dnd_count,
        conflict_count,
        voice_count,
    )

    for c in all_chunks:
        if c.do_not_discuss:
            log.info(
                "  [DND %-30s]  %s  %.70s",
                c.do_not_discuss_mode,
                c.chunk_id,
                c.content.replace("\n", " "),
            )
    for c in all_chunks:
        if c.corpus_conflict:
            log.info(
                "  [CONFLICT]  %s  %.70s", c.chunk_id, c.content.replace("\n", " ")
            )

    if args.dry_run:
        log.info("Dry run — skipping embedding and DB write.")
        return

    # ── Embed + persist ───────────────────────────────────────────────────────
    embedder = GeminiEmbedder()
    conn = init_db(args.db)

    texts = [c.content for c in all_chunks]
    log.info("Embedding %d chunks via Gemini text-embedding-004 ...", len(texts))
    embeddings = embedder.embed(texts)

    for chunk, emb in zip(all_chunks, embeddings):
        insert_chunk(conn, chunk, emb)

    conn.commit()
    conn.close()

    log.info("Done. %d chunks written to %s", len(all_chunks), args.db)


if __name__ == "__main__":
    main()
