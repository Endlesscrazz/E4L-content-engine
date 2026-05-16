"""
Empirical comparison: Opus 4.7 vs Sonnet 4.6 on the same certainty-inflation draft.

Usage:
    python scripts/compare_validator_models.py

Requires ANTHROPIC_API_KEY in environment (.env or exported).
Results should be recorded in DECISIONS.md and METHODOLOGY.md.

Staged scenario: source corpus says "may support energy levels" (hedged modal).
The draft claims "restores your energy within days" (guarantee + timeframe).
Both models should detect the inflation; the question is whether they differ
in precision, false positives, or revision note quality.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure ANTHROPIC_API_KEY is set before importing clients
if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")

from app.clients import AnthropicClient
from app.models import (
    Audience,
    Claim,
    ClaimTaxonomy,
    ContentBrief,
    Draft,
    FunnelStage,
    Platform,
    SourceChunk,
    ToneRegister,
)
from app.validator_llm import run_llm_judge

# ─── Staged fixtures ───────────────────────────────────────────────────────────
#
# Source chunk: hedged language ("may support", "some clients report").
# Draft: certainty-inflated version ("restores", "you will feel", guaranteed timeframe).
# A good judge catches the inflation; a weak judge lets it through.

SOURCE_CHUNK = SourceChunk(
    chunk_id="e4l_ai_energy_001",
    doc_name="AI_Version_Restore_Your_Energy.docx",
    content=(
        "The NES Health BioWellness Scanner may support the body's natural energy "
        "regulation systems by detecting and addressing distortions in the bioenergetic "
        "field. Some clients report noticing improved energy levels and reduced fatigue "
        "over time. Preliminary findings suggest that consistent use of the miHealth "
        "device, alongside Infoceuticals, is associated with a sense of improved "
        "wellbeing for many users."
    ),
    doc_type="concept_explainer",
    is_voice_anchor=False,
)

VOICE_ANCHOR = SourceChunk(
    chunk_id="e4l_ai_voice_001",
    doc_name="AI_Version_Restore_Your_Energy.docx",
    content=(
        "Think of the body like a sophisticated electrical system. When the wiring is "
        "clean and the signal is strong, everything works as it should. But when there "
        "is interference — what we call 'distortion' in the bioenergetic field — the "
        "system starts to compensate, and that compensation costs energy. "
        "Harry Massey's work, drawing on Gerald Pollack's research into structured "
        "water and Fritz-Albert Popp's discoveries in biophotonics, suggests that "
        "restoring field integrity is the upstream intervention that makes everything "
        "else downstream more effective."
    ),
    doc_type="concept_explainer",
    is_voice_anchor=True,
)

# The draft inflates certainty in three ways:
#   1. "restores your energy" — source says "may support energy regulation"
#   2. "you will feel" — source says "some clients report"
#   3. "proven" — source says "preliminary findings suggest"
INFLATED_DRAFT = Draft(
    platform=Platform.LINKEDIN,
    body=(
        "Struggling with energy that just won't come back no matter what you try? "
        "The NES Health BioWellness Scanner restores your energy by fixing the root "
        "cause — distortions in your bioenergetic field. Once your field is clear, "
        "you will feel the difference within days. Our technology, proven to improve "
        "energy and reduce fatigue, is the upstream intervention your body has been "
        "waiting for. Stop managing symptoms. Start restoring your field."
    ),
    claims=[
        Claim(
            text="restores your energy by fixing the root cause",
            chunk_id=SOURCE_CHUNK.chunk_id,
            cited_substring="may support the body's natural energy regulation systems",
            taxonomy=ClaimTaxonomy.B,
            certainty_inflation=True,
        ),
        Claim(
            text="you will feel the difference within days",
            chunk_id=SOURCE_CHUNK.chunk_id,
            cited_substring="Some clients report noticing improved energy levels",
            taxonomy=ClaimTaxonomy.B,
            certainty_inflation=True,
        ),
        Claim(
            text="proven to improve energy and reduce fatigue",
            chunk_id=SOURCE_CHUNK.chunk_id,
            cited_substring="Preliminary findings suggest",
            taxonomy=ClaimTaxonomy.B,
            certainty_inflation=True,
        ),
    ],
    cited_chunk_ids=[SOURCE_CHUNK.chunk_id],
)

BRIEF = ContentBrief(
    goal="Drive awareness of NES Health products for people with chronic fatigue.",
    audience=Audience.CONSUMER,
    funnel_stage=FunnelStage.COLD,
    tone=ToneRegister.CONVERSATIONAL,
    topic_focus="chronic fatigue",
)

MODELS = [
    ("claude-opus-4-7", "Opus 4.7"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
]


def run_comparison() -> None:
    client = AnthropicClient()
    results: list[dict] = []

    print("\n" + "=" * 70)
    print("VALIDATOR MODEL COMPARISON — CERTAINTY INFLATION DETECTION")
    print("=" * 70)
    print(f"\nDraft platform : {INFLATED_DRAFT.platform.value}")
    print(f"Known inflations: {len([c for c in INFLATED_DRAFT.claims if c.certainty_inflation])}")
    print()

    for model_id, model_label in MODELS:
        print(f"── {model_label} ({model_id}) ──")
        t0 = time.perf_counter()
        result, tokens_in, tokens_out, cost = run_llm_judge(
            draft=INFLATED_DRAFT,
            cited_chunks=[SOURCE_CHUNK],
            brief=BRIEF,
            voice_anchors=[VOICE_ANCHOR],
            client=client,
            model=model_id,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        print(f"  passed                    : {result.passed}")
        print(f"  grounding_pass            : {result.grounding_pass}")
        print(f"  certainty_inflation_issues: {result.certainty_inflation_issues}")
        print(f"  taxonomy_issues           : {result.taxonomy_issues}")
        print(f"  voice_pass                : {result.voice_pass}")
        print(f"  tone_pass                 : {result.tone_pass}")
        print(f"  revision_notes            : {result.revision_notes}")
        print(f"  tokens_in / tokens_out    : {tokens_in} / {tokens_out}")
        print(f"  cost_usd                  : ${cost:.6f}")
        print(f"  latency_ms                : {latency_ms}")
        print()

        results.append({
            "model": model_label,
            "model_id": model_id,
            "passed": result.passed,
            "inflation_caught": len(result.certainty_inflation_issues),
            "taxonomy_issues": len(result.taxonomy_issues),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": cost,
            "latency_ms": latency_ms,
            "revision_notes": result.revision_notes,
        })

    print("=" * 70)
    print("SUMMARY — record these numbers in DECISIONS.md + METHODOLOGY.md")
    print("=" * 70)
    print(f"{'Model':<14} {'Passed':<8} {'Inflation caught':<18} {'Tokens in':<12} {'Cost USD':<12} {'Latency ms'}")
    for r in results:
        print(
            f"{r['model']:<14} {str(r['passed']):<8} {r['inflation_caught']:<18} "
            f"{r['tokens_in']:<12} ${r['cost_usd']:<11.6f} {r['latency_ms']}"
        )
    print()
    print("Record your observations below and paste into DECISIONS.md:")
    print("  - Did Opus catch all 3 inflations? Did Sonnet?")
    print("  - Were revision notes actionable (specific phrase + fix direction)?")
    print("  - Cost delta: Opus is ~5x more expensive per output token.")
    print("  - Was the quality difference worth it for this check type?")


if __name__ == "__main__":
    run_comparison()
