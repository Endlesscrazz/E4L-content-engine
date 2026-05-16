"""
Full Validator pipeline — assembles all four layers into a single validate_draft() call.

Pipeline order (short-circuits on first deterministic failure):
  Layer 0  check_editorial       — lexicon/regex, no API cost
  Layer 1  check_citations       — substring resolution, no API cost
  Layer 4  check_do_not_discuss  — citation join + body scan, no API cost (runs 3rd)
  Layers 2+3  run_llm_judge      — Opus 4.7, only reached if L0/L1/L4 all pass

Short-circuit design: when a gate fires, downstream gates are skipped and marked
with sentinel results. The top-level verdict.revision_notes carries the first
gate's failure message — that's what the Writer needs to act on.
"""

from __future__ import annotations

from app.clients import AnthropicClient
from app.models import (
    CitationCheckResult,
    ContentBrief,
    DoNotDiscussCheckResult,
    Draft,
    SourceChunk,
    ValidatorVerdict,
)
from app.validator_gates import DndFlag, check_citations, check_do_not_discuss, check_editorial
from app.validator_llm import VALIDATOR_MODEL, run_llm_judge


def validate_draft(
    draft: Draft,
    corpus_chunks: dict[str, SourceChunk],
    brief: ContentBrief,
    dnd_flags: list[DndFlag],
    voice_anchors: list[SourceChunk],
    client: AnthropicClient,
    model: str = VALIDATOR_MODEL,
) -> ValidatorVerdict:
    """Run all four Validator layers in order; short-circuit on first failure.

    corpus_chunks: pre-fetched by caller (Orchestrator) — Validator is stateless w.r.t. DB.
    voice_anchors: pre-fetched by caller — consistent with cited_chunks pattern (S3 decision).
    dnd_flags: loaded from do_not_discuss.yaml by caller before this call.
    """
    # Layer 0 — editorial pre-gate (no API cost, always runs first)
    editorial = check_editorial(draft)
    if not editorial.passed:
        return ValidatorVerdict(
            draft_platform=draft.platform,
            editorial=editorial,
            citations=_skipped_citations(),
            do_not_discuss=_skipped_dnd(),
            llm_judge=None,
            revision_notes=editorial.message,
        ).compute_ship()

    # Layer 1 — citation resolution (no API cost)
    citations = check_citations(draft, corpus_chunks)
    if not citations.passed:
        return ValidatorVerdict(
            draft_platform=draft.platform,
            editorial=editorial,
            citations=citations,
            do_not_discuss=_skipped_dnd(),
            llm_judge=None,
            revision_notes=citations.message,
        ).compute_ship()

    # Layer 4 (runs 3rd) — do_not_discuss gate (no API cost)
    # Resolve cited_chunk_ids → SourceChunk objects for the dnd citation-join pass.
    cited_chunks = [corpus_chunks[cid] for cid in draft.cited_chunk_ids if cid in corpus_chunks]
    dnd = check_do_not_discuss(draft, cited_chunks, dnd_flags, brief)
    if not dnd.passed:
        return ValidatorVerdict(
            draft_platform=draft.platform,
            editorial=editorial,
            citations=citations,
            do_not_discuss=dnd,
            llm_judge=None,
            revision_notes=dnd.message,
        ).compute_ship()

    # Layers 2+3 — Opus 4.7 LLM judge (only reached if L0/L1/L4 all pass)
    llm_result, tokens_in, tokens_out, cost = run_llm_judge(
        draft=draft,
        cited_chunks=cited_chunks,
        brief=brief,
        voice_anchors=voice_anchors,
        client=client,
        model=model,
    )

    return ValidatorVerdict(
        draft_platform=draft.platform,
        editorial=editorial,
        citations=citations,
        do_not_discuss=dnd,
        llm_judge=llm_result,
        revision_notes=llm_result.revision_notes if not llm_result.passed else None,
        model_used=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
    ).compute_ship()


# ─── Skipped-layer sentinels ───────────────────────────────────────────────────
# When a gate short-circuits the pipeline, downstream layers are not run.
# Sentinels make this explicit in the verdict (passed=False + message="skipped").
# compute_ship() still produces ship=False correctly because the triggering gate
# is also passed=False — the AND chain fails at the first failure.

def _skipped_citations() -> CitationCheckResult:
    return CitationCheckResult(passed=False, message="skipped — upstream gate failed")


def _skipped_dnd() -> DoNotDiscussCheckResult:
    return DoNotDiscussCheckResult(passed=False, message="skipped — upstream gate failed")
