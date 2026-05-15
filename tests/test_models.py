"""
Smoke tests for pydantic models — S0 testable criterion.
No LLM calls, no I/O.
"""

import pytest
from app.models import (
    Audience,
    ClaimTaxonomy,
    ContentBrief,
    Draft,
    FunnelStage,
    Platform,
    RunBudget,
    ValidatorVerdict,
    EditorialCheckResult,
    CitationCheckResult,
    DoNotDiscussCheckResult,
    LLMJudgeResult,
)


def test_content_brief_defaults():
    brief = ContentBrief(goal="test goal")
    assert brief.audience == Audience.CONSUMER
    assert brief.funnel_stage == FunnelStage.COLD
    assert Platform.LINKEDIN in brief.platforms
    assert Platform.EMAIL in brief.platforms


def test_content_brief_validation():
    with pytest.raises(Exception):
        ContentBrief(goal="")  # min_length=5


def test_draft_email_fields():
    d = Draft(
        platform=Platform.EMAIL,
        subject="Test subject",
        body="Test body",
    )
    assert d.subject == "Test subject"
    assert d.cost_usd == 0.0


def test_run_budget_exhaustion():
    budget = RunBudget(max_turns=2, max_cost_usd=0.10)
    assert not budget.is_exhausted()
    budget.record(0.04)
    budget.record(0.04)
    assert budget.turns_remaining() == 0
    assert budget.is_exhausted()


def test_run_budget_cost_cap():
    budget = RunBudget(max_turns=100, max_cost_usd=0.10)
    budget.record(0.11)
    assert budget.is_exhausted()


def test_validator_verdict_ship_requires_all_pass():
    verdict = ValidatorVerdict(
        draft_platform=Platform.LINKEDIN,
        editorial=EditorialCheckResult(passed=True),
        citations=CitationCheckResult(passed=True),
        do_not_discuss=DoNotDiscussCheckResult(passed=True),
        llm_judge=LLMJudgeResult(passed=True),
    )
    verdict.compute_ship()
    assert verdict.ship is True


def test_validator_verdict_editorial_fail_no_ship():
    verdict = ValidatorVerdict(
        draft_platform=Platform.LINKEDIN,
        editorial=EditorialCheckResult(passed=False, triggered_phrase="cure all disease"),
        citations=CitationCheckResult(passed=True),
        do_not_discuss=DoNotDiscussCheckResult(passed=True),
        llm_judge=LLMJudgeResult(passed=True),
    )
    verdict.compute_ship()
    assert verdict.ship is False


def test_validator_verdict_no_llm_judge_no_ship():
    verdict = ValidatorVerdict(
        draft_platform=Platform.EMAIL,
        editorial=EditorialCheckResult(passed=True),
        citations=CitationCheckResult(passed=True),
        do_not_discuss=DoNotDiscussCheckResult(passed=True),
        llm_judge=None,
    )
    verdict.compute_ship()
    assert verdict.ship is False
