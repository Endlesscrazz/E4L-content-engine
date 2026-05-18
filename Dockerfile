# Pin to linux/amd64: sqlite-vec 0.1.6 has no Linux arm64 wheel (32-bit fallback
# causes ELFCLASS32 error). amd64 runs via Rosetta 2 on Apple Silicon at near-native
# speed, and matches the evaluator's Linux x86_64 environment.
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Install system deps needed by python-docx (lxml) and sqlite-vec
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ app/
COPY agents/ agents/
COPY prompts/ prompts/
COPY scripts/ scripts/
COPY static/ static/
# Corpus: YAML config files (annotations + denylist) are committed and needed by ingest.
# corpus.db is NOT committed — entrypoint generates it on first boot.
COPY corpus/ corpus/

# Source documents (6 E4L docs) — committed to repo, needed by ingest script
COPY source_docs/ source_docs/

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
