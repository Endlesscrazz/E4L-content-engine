"""
Resilient client wrappers for Anthropic, Gemini, and Brave.

Each wrapper retries on 429 / transient errors with exponential backoff + jitter.
Keys are loaded from environment only — never passed as arguments, never logged.

Backoff policy: up to 5 attempts, base delay 1s, max 60s, full jitter.
Considered a fixed-delay loop: rejected — thundering herd under rate limits.
Considered tenacity: rejected — adds a dependency; the backoff logic here is
  small enough to own directly, and the retry conditions differ per API.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

import anthropic
import google.generativeai as genai
import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─── Backoff helper ───────────────────────────────────────────────────────────

_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 60.0

ANTHROPIC_INPUT_COST = {
    "claude-haiku-4-5-20251001": 0.80 / 1_000_000,
    "claude-sonnet-4-6": 3.00 / 1_000_000,
    "claude-opus-4-7": 15.00 / 1_000_000,
}
ANTHROPIC_OUTPUT_COST = {
    "claude-haiku-4-5-20251001": 4.00 / 1_000_000,
    "claude-sonnet-4-6": 15.00 / 1_000_000,
    "claude-opus-4-7": 75.00 / 1_000_000,
}


def _jitter_delay(attempt: int) -> float:
    """Full-jitter exponential backoff — avoids synchronized retries."""
    cap = min(_MAX_DELAY_S, _BASE_DELAY_S * (2 ** attempt))
    return random.uniform(0, cap)


# ─── Anthropic ────────────────────────────────────────────────────────────────

class AnthropicClient:
    """
    Thin wrapper around the Anthropic Messages API.
    Adds retry-on-429, per-call cost accounting, and structured error logging.
    Does NOT hide the raw message format — callers build messages themselves
    so orchestration logic stays transparent.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        # Synchronous client; async calls use asyncio.to_thread in the agents.
        self._client = anthropic.Anthropic(api_key=api_key)

    def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> tuple[anthropic.types.Message, float]:
        """
        Returns (Message, cost_usd).
        Retries on 429 with backoff; raises on other errors after logging.
        """
        for attempt in range(_MAX_ATTEMPTS):
            try:
                kwargs: dict[str, Any] = dict(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if system:
                    kwargs["system"] = system
                if tools:
                    kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

                response = self._client.messages.create(**kwargs)
                cost = self._compute_cost(model, response)
                return response, cost

            except anthropic.RateLimitError:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                delay = _jitter_delay(attempt)
                logger.warning("Anthropic 429 — attempt %d/%d, sleeping %.1fs", attempt + 1, _MAX_ATTEMPTS, delay)
                time.sleep(delay)

            except anthropic.APIStatusError as exc:
                # Non-429 API errors are not retried — surface immediately.
                logger.error("Anthropic API error %s: %s", exc.status_code, exc.message)
                raise

    @staticmethod
    def _compute_cost(model: str, response: anthropic.types.Message) -> float:
        in_tokens = response.usage.input_tokens
        out_tokens = response.usage.output_tokens
        in_rate = ANTHROPIC_INPUT_COST.get(model, 0.0)
        out_rate = ANTHROPIC_OUTPUT_COST.get(model, 0.0)
        return round(in_tokens * in_rate + out_tokens * out_rate, 8)


# ─── Gemini (embeddings only) ─────────────────────────────────────────────────

class GeminiEmbedder:
    """
    Wraps the Gemini text-embedding-004 model (3072 dimensions).
    Only used by the ingestion script — not in the hot path.
    Retries on quota/503 with backoff.
    """

    MODEL = "models/text-embedding-004"
    DIMS = 3072

    def __init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts. Returns a list of 3072-dim float vectors."""
        results: list[list[float]] = []
        for text in texts:
            for attempt in range(_MAX_ATTEMPTS):
                try:
                    result = genai.embed_content(
                        model=self.MODEL,
                        content=text,
                        task_type="retrieval_document",
                    )
                    results.append(result["embedding"])
                    break
                except Exception as exc:
                    if attempt == _MAX_ATTEMPTS - 1:
                        raise
                    delay = _jitter_delay(attempt)
                    logger.warning("Gemini embed error %s — retrying in %.1fs", exc, delay)
                    time.sleep(delay)
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (retrieval_query task type)."""
        for attempt in range(_MAX_ATTEMPTS):
            try:
                result = genai.embed_content(
                    model=self.MODEL,
                    content=text,
                    task_type="retrieval_query",
                )
                return result["embedding"]
            except Exception as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                delay = _jitter_delay(attempt)
                logger.warning("Gemini query embed error %s — retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError("embed_query failed after all retries")  # unreachable


# ─── Brave Search ─────────────────────────────────────────────────────────────

class BraveSearchResult(BaseModel):
    title: str
    url: str
    description: str


class BraveClient:
    """
    Thin wrapper around the Brave Search API.
    Retries on 429 / 503. Returns structured results or empty list on failure
    so the Researcher's graceful-empty path kicks in without an exception.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self) -> None:
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            raise EnvironmentError("BRAVE_API_KEY is not set")
        self._api_key = api_key

    def search(self, query: str, count: int = 5) -> list[BraveSearchResult]:
        """Returns up to `count` web results. Returns [] if quota hit after retries."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {"q": query, "count": count}

        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = requests.get(
                    self.BASE_URL,
                    headers=headers,
                    params=params,
                    timeout=10,
                )
                if resp.status_code == 429:
                    if attempt == _MAX_ATTEMPTS - 1:
                        logger.warning("Brave 429 — returning empty results after %d attempts", _MAX_ATTEMPTS)
                        return []
                    delay = _jitter_delay(attempt)
                    logger.warning("Brave 429 — attempt %d/%d, sleeping %.1fs", attempt + 1, _MAX_ATTEMPTS, delay)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                results = data.get("web", {}).get("results", [])
                return [
                    BraveSearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        description=r.get("description", ""),
                    )
                    for r in results
                ]

            except requests.exceptions.RequestException as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    logger.error("Brave search failed: %s — returning empty", exc)
                    return []
                delay = _jitter_delay(attempt)
                logger.warning("Brave error %s — retrying in %.1fs", exc, delay)
                time.sleep(delay)

        return []

    def read_url(self, url: str) -> str:
        """Fetch plain text from a URL. Returns '' on failure — never raises."""
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "E4L-ContentEngine/1.0"})
            resp.raise_for_status()
            # Basic HTML stripping — caller can do further cleaning.
            from html.parser import HTMLParser

            class _Stripper(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self._parts: list[str] = []

                def handle_data(self, data: str) -> None:
                    self._parts.append(data)

                def get_text(self) -> str:
                    return " ".join(self._parts)

            parser = _Stripper()
            parser.feed(resp.text)
            return parser.get_text()[:8000]  # cap at 8K chars to stay within context budget
        except Exception as exc:
            logger.warning("read_url %s failed: %s", url, exc)
            return ""
