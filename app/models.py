"""
Pydantic contracts at every agent boundary.
All inter-agent data crosses through these types — no raw dicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class Platform(str, Enum):
    LINKEDIN = "linkedin"
    EMAIL = "email"


class Audience(str, Enum):
    CONSUMER = "consumer"
    PRACTITIONER = "practitioner"


class FunnelStage(str, Enum):
    COLD = "cold"
    WARM = "warm"
    HOT = "hot"
    CUSTOMER = "customer"


class ToneRegister(str, Enum):
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"
    CLINICAL = "clinical"
    INSPIRATIONAL = "inspirational"


class ClaimTaxonomy(str, Enum):
    # A: direct paraphrase of source — must cite
    A = "A"
    # B: implied by source but not stated — must cite + flag as inference
    B = "B"
    # C: general knowledge, not E4L-specific — no citation required
    C = "C"
    # D: novel claim not in source — HARD FAIL
    D = "D"


# ─── ContentBrief ─────────────────────────────────────────────────────────────

class ContentBrief(BaseModel):
    """User-facing request. Every axis is a typed knob; each maps to a retrieval
    filter, Writer prompt section, or voice anchor selector — not just flavor."""

    goal: str = Field(
        ...,
        description="Plain-language description of what the content should achieve.",
        min_length=5,
        max_length=500,
    )
    audience: Audience = Audience.CONSUMER
    funnel_stage: FunnelStage = FunnelStage.COLD
    platforms: list[Platform] = Field(
        default_factory=lambda: [Platform.LINKEDIN, Platform.EMAIL]
    )
    topic_focus: str | None = Field(
        default=None,
        description="Optional topic constraint (e.g. 'chronic fatigue', 'sleep').",
    )
    product_focus: list[str] = Field(
        default_factory=list,
        description="Specific E4L products to foreground (e.g. ['miHealth', 'BWS']).",
    )
    tone: ToneRegister = ToneRegister.CONVERSATIONAL
    format_intent: str | None = Field(
        default=None,
        description="Optional structural hint (e.g. 'story-first', 'research-led').",
    )


# ─── Corpus types ─────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """A retrieved corpus chunk passed to Writers and Validator.
    chunk_id is the stable reference for citations."""

    chunk_id: str
    doc_name: str
    content: str
    doc_type: str
    audience_tags: list[str] = Field(default_factory=list)
    product_associations: list[str] = Field(default_factory=list)
    is_voice_anchor: bool = False
    do_not_discuss: bool = False
    do_not_discuss_mode: Literal["never_in_generated_content", "do_not_volunteer"] | None = None
    corpus_conflict: bool = False


# ─── Claim ────────────────────────────────────────────────────────────────────

class Claim(BaseModel):
    """A single claim extracted from a draft, with citation details."""

    text: str = Field(..., description="The exact claim text as it appears in the draft.")
    chunk_id: str | None = Field(
        default=None,
        description="The chunk_id this claim cites. None for Type-C uncited claims.",
    )
    cited_substring: str | None = Field(
        default=None,
        description="The exact substring from the cited chunk supporting this claim.",
    )
    taxonomy: ClaimTaxonomy | None = None
    certainty_inflation: bool = False


# ─── Draft ────────────────────────────────────────────────────────────────────

class Draft(BaseModel):
    """Writer output — one platform's content with its cited claims."""

    platform: Platform
    # For LinkedIn: a single body field.
    # For Email: subject + body both required.
    subject: str | None = Field(default=None, description="Email subject line. Required if platform=email.")
    body: str = Field(..., description="Full draft body text.")
    claims: list[Claim] = Field(
        default_factory=list,
        description="Structured claims extracted inline by the Writer.",
    )
    # The Writer indicates which chunks it used so Validator can target them.
    cited_chunk_ids: list[str] = Field(default_factory=list)
    model_used: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


# ─── ValidatorVerdict ─────────────────────────────────────────────────────────

class EditorialCheckResult(BaseModel):
    """Result of the deterministic pre-gate (Layer 0).
    Runs before any LLM call — fully unit-testable."""

    passed: bool
    # Exact regex/phrase that triggered a failure.
    triggered_phrase: str | None = None
    message: str | None = None


class CitationCheckResult(BaseModel):
    """Result of the deterministic citation-resolution gate (Layer 1)."""

    passed: bool
    failed_claims: list[str] = Field(
        default_factory=list,
        description="claim.text values where the cited_substring was not found in the chunk.",
    )
    message: str | None = None


class DoNotDiscussCheckResult(BaseModel):
    """Result of the mode-aware do_not_discuss gate (Layer 4).
    Two-pass: citation-chunk join + draft-body topic scan."""

    passed: bool
    triggered_flag: str | None = None
    triggered_mode: str | None = None
    message: str | None = None


class LLMJudgeResult(BaseModel):
    """Result of the Opus 4.7 LLM judge (Layers 2+3 combined).
    Grounding, claim taxonomy, certainty-inflation, voice, tone."""

    passed: bool
    grounding_pass: bool = True
    taxonomy_issues: list[str] = Field(
        default_factory=list,
        description="Type-D claim texts that caused failure.",
    )
    certainty_inflation_issues: list[str] = Field(
        default_factory=list,
        description="Claims where asserted certainty exceeds source-supported certainty.",
    )
    voice_pass: bool = True
    tone_pass: bool = True
    revision_notes: str | None = None


class ValidatorVerdict(BaseModel):
    """Final Validator output. Ship only if all five checks pass."""

    draft_platform: Platform
    # Layer 0 — deterministic editorial pre-gate
    editorial: EditorialCheckResult
    # Layer 1 — deterministic citation resolution
    citations: CitationCheckResult
    # Layer 4 — deterministic do_not_discuss (run 3rd so we short-circuit early)
    do_not_discuss: DoNotDiscussCheckResult
    # Layers 2+3 — LLM judge (only reached if L0/L1/L4 all pass)
    llm_judge: LLMJudgeResult | None = None

    ship: bool = False
    revision_notes: str | None = None
    model_used: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def compute_ship(self) -> "ValidatorVerdict":
        """Set ship=True only if every gate passed."""
        deterministic_ok = (
            self.editorial.passed
            and self.citations.passed
            and self.do_not_discuss.passed
        )
        llm_ok = self.llm_judge.passed if self.llm_judge else False
        self.ship = deterministic_ok and llm_ok
        return self


# ─── Observability ────────────────────────────────────────────────────────────

class AgentTelemetry(BaseModel):
    """Per-agent cost + latency record written to observability sqlite."""

    trace_id: str
    agent_name: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    tool_calls: int = 0
    error: str | None = None


# ─── Researcher ───────────────────────────────────────────────────────────────

class ResearchFinding(BaseModel):
    """One scored web result kept by the Researcher (label keep or weak)."""

    title: str
    url: str
    snippet: str = Field(..., description="Scored text excerpt, capped at 600 chars.")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    relevance_label: Literal["keep", "weak", "drop"]


class ResearchResult(BaseModel):
    """Full output of a Researcher run. findings=[] when nothing passes threshold."""

    findings: list[ResearchFinding] = Field(default_factory=list)
    no_context_reason: str | None = Field(
        default=None,
        description="Set to 'no trend context available' when Brave returns nothing "
                    "useful or all results score below threshold after reformulation.",
    )
    actions_used: int = 0
    cost_usd: float = 0.0


# ─── RunBudget ────────────────────────────────────────────────────────────────

class RunBudget(BaseModel):
    """Code-side enforcement of caps.
    The Orchestrator checks this object before every dispatch —
    caps are not in the system prompt, they are in Python."""

    # Hard limits
    max_turns: int = 15
    max_cost_usd: float = 0.50

    # Running state (mutated during a run)
    turns_used: int = 0
    cost_usd_spent: float = 0.0

    def turns_remaining(self) -> int:
        return self.max_turns - self.turns_used

    def cost_remaining(self) -> float:
        return round(self.max_cost_usd - self.cost_usd_spent, 6)

    def is_exhausted(self) -> bool:
        return self.turns_used >= self.max_turns or self.cost_usd_spent >= self.max_cost_usd

    def record(self, cost_usd: float) -> None:
        self.turns_used += 1
        self.cost_usd_spent = round(self.cost_usd_spent + cost_usd, 6)
