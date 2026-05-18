#!/usr/bin/env bash
# First-boot corpus ingest guard.
# corpus/corpus.db is gitignored and not in the image; generate it on first start.
# On subsequent starts (stop → start without --build) the db already exists, so skip.
set -euo pipefail

if [ ! -f /app/corpus/corpus.db ]; then
    echo "[entrypoint] corpus.db not found — running first-boot ingest (takes ~60s)..."
    python scripts/ingest_corpus.py source_docs/
    echo "[entrypoint] ingest complete."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
