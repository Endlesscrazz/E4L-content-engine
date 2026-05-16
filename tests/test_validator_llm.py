"""
Tests for the Validator LLM judge and full pipeline assembler (validate_draft).

Invariant: no real Anthropic API calls anywhere in this file.
The LLM judge is tested via unittest.mock.patch on app.validator.run_llm_judge.

What's covered:
  - validate_draft short-circuits on each deterministic gate failure (3 cases)
  - LLM judge is called only when all three gates pass
  - ship=True when LLM judge passes
  - ship=False when LLM judge fails, revision_notes propagated
  - _parse_verdict maps tool call args to LLMJudgeResult correctly
  - validate_draft uses model kwarg when provided (passed through to LLM judge)
"""

from unittest.mock import MagicMock, patch

import pytest

from app.models import (
    Audience,
    Claim,
    ClaimTaxonomy,
    ContentBrief,
    Draft,
    FunnelStage,
    LLMJudgeResult,
    Platform,
    SourceChunk,
    ToneRegister,
)
from app.validator import validate_draft
from app.validator_gates import DndFlag
from app.validator_llm import _parse_verdict


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def corpus_chunk() -> SourceChunk:
    return SourceChunk(
        chunk_id="e4l_ai_001",
        doc_name="AI_Version.docx",
        content="The bioenergetic field may support healthy cellular function and energy levels.",
        doc_type="concept_explainer",
    )


@pytest.fixture
def corpus(corpus_chunk) -> dict[str, SourceChunk]:
    return {corpus_chunk.chunk_id: corpus_chunk}


@pytest.fixture
def brief() -> ContentBrief:
    return ContentBrief(
        goal="Explain how bioenergetics may support energy.",
        audience=Audience.CONSUMER,
        funnel_stage=FunnelStage.COLD,
        tone=ToneRegister.CONVERSATIONAL,
    )


@pytest.fixture
def clean_draft(corpus_chunk) -> Draft:
    """Draft that passes all three deterministic gates."""
    return Draft(
        platform=Platform.LINKEDIN,
        body="Research suggests that the bioenergetic field may support healthy cellular function.",
        claims=[
            Claim(
                text="the bioenergetic field may support healthy cellular function",
                chunk_id=corpus_chunk.chunk_id,
                cited_substring="bioenergetic field may support healthy cellular function",
                taxonomy=ClaimTaxonomy.A,
            )
        ],
        cited_chunk_ids=[corpus_chunk.chunk_id],
    )


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


# ─── _parse_verdict unit tests ─────────────────────────────────────────────────

class TestParseVerdict:
    def test_parse_pass(self):
        result = _parse_verdict({
            "grounding_pass": True,
            "taxonomy_issues": [],
            "certainty_inflation_issues": [],
            "voice_pass": True,
            "tone_pass": True,
            "passed": True,
        })
        assert result.passed is True
        assert result.grounding_pass is True
        assert result.taxonomy_issues == []
        assert result.certainty_inflation_issues == []
        assert result.voice_pass is True
        assert result.tone_pass is True
        assert result.revision_notes is None

    def test_parse_fail_with_taxonomy_issue(self):
        result = _parse_verdict({
            "grounding_pass": False,
            "taxonomy_issues": ["BWS eliminates chronic fatigue permanently."],
            "certainty_inflation_issues": [],
            "voice_pass": True,
            "tone_pass": True,
            "passed": False,
            "revision_notes": "Type-D claim: 'BWS eliminates chronic fatigue permanently' has no source support.",
        })
        assert result.passed is False
        assert result.taxonomy_issues == ["BWS eliminates chronic fatigue permanently."]
        assert result.revision_notes is not None
        assert "Type-D" in result.revision_notes

    def test_parse_fail_with_certainty_inflation(self):
        result = _parse_verdict({
            "grounding_pass": True,
            "taxonomy_issues": [],
            "certainty_inflation_issues": ["BWS restores your energy within days."],
            "voice_pass": True,
            "tone_pass": True,
            "passed": False,
            "revision_notes": "Certainty inflation: source says 'may support', draft says 'restores'.",
        })
        assert result.passed is False
        assert result.certainty_inflation_issues == ["BWS restores your energy within days."]

    def test_parse_pass_clears_revision_notes(self):
        """revision_notes must be None when passed=True even if the field was populated."""
        result = _parse_verdict({
            "grounding_pass": True,
            "taxonomy_issues": [],
            "certainty_inflation_issues": [],
            "voice_pass": True,
            "tone_pass": True,
            "passed": True,
            "revision_notes": "Some stray text.",  # model shouldn't do this, but we guard it
        })
        assert result.passed is True
        assert result.revision_notes is None

    def test_parse_missing_optional_lists(self):
        """taxonomy_issues and certainty_inflation_issues default to [] if absent."""
        result = _parse_verdict({
            "grounding_pass": True,
            "voice_pass": True,
            "tone_pass": True,
            "passed": True,
        })
        assert result.taxonomy_issues == []
        assert result.certainty_inflation_issues == []


# ─── validate_draft short-circuit tests ──────────────────────────────────────

class TestValidateDraftShortCircuit:
    """LLM judge must NOT be called when any deterministic gate fires."""

    def test_editorial_fail_no_llm_call(self, corpus, brief, mock_client):
        draft = Draft(
            platform=Platform.LINKEDIN,
            body="This protocol cures chronic fatigue syndrome completely.",
            claims=[],
            cited_chunk_ids=[],
        )
        with patch("app.validator.run_llm_judge") as mock_judge:
            verdict = validate_draft(draft, corpus, brief, [], [], mock_client)
            mock_judge.assert_not_called()

        assert not verdict.ship
        assert not verdict.editorial.passed
        assert "cure" in verdict.editorial.triggered_phrase.lower()
        assert verdict.citations.message == "skipped — upstream gate failed"
        assert verdict.do_not_discuss.message == "skipped — upstream gate failed"
        assert verdict.llm_judge is None

    def test_citation_fail_no_llm_call(self, corpus, brief, mock_client):
        draft = Draft(
            platform=Platform.LINKEDIN,
            body="The bioenergetic field may support healthy cellular function.",
            claims=[
                Claim(
                    text="bioenergetic field claim",
                    chunk_id="e4l_ai_001",
                    cited_substring="THIS SUBSTRING DOES NOT EXIST IN THE CHUNK",
                    taxonomy=ClaimTaxonomy.A,
                )
            ],
            cited_chunk_ids=["e4l_ai_001"],
        )
        with patch("app.validator.run_llm_judge") as mock_judge:
            verdict = validate_draft(draft, corpus, brief, [], [], mock_client)
            mock_judge.assert_not_called()

        assert not verdict.ship
        assert verdict.editorial.passed
        assert not verdict.citations.passed
        assert verdict.llm_judge is None

    def test_dnd_fail_no_llm_call(self, corpus, brief, mock_client):
        """Draft that cites a chunk flagged never_in_generated_content."""
        flagged_chunk = SourceChunk(
            chunk_id="peter_fraser_001",
            doc_name="Origin_Story.docx",
            content="Peter Fraser passed away in 2022.",
            doc_type="origin_story",
            do_not_discuss=True,
            do_not_discuss_mode="never_in_generated_content",
        )
        flagged_corpus = {**corpus, "peter_fraser_001": flagged_chunk}
        dnd_flag = DndFlag(
            flag="peter_fraser_death",
            mode="never_in_generated_content",
            trigger_regex=r"\bPeter\s+Fraser\b",
        )
        draft = Draft(
            platform=Platform.LINKEDIN,
            body="In memory of our founder, Peter Fraser's work lives on.",
            claims=[],
            cited_chunk_ids=["peter_fraser_001"],
        )
        with patch("app.validator.run_llm_judge") as mock_judge:
            verdict = validate_draft(draft, flagged_corpus, brief, [dnd_flag], [], mock_client)
            mock_judge.assert_not_called()

        assert not verdict.ship
        assert verdict.editorial.passed
        assert verdict.citations.passed
        assert not verdict.do_not_discuss.passed
        assert verdict.llm_judge is None


# ─── validate_draft LLM judge integration ─────────────────────────────────────

class TestValidateDraftLLMJudge:
    """LLM judge is mocked — tests pipeline routing and verdict assembly only."""

    def _make_llm_result(self, passed: bool, notes: str | None = None) -> LLMJudgeResult:
        return LLMJudgeResult(
            passed=passed,
            grounding_pass=passed,
            taxonomy_issues=[] if passed else ["some Type-D claim"],
            certainty_inflation_issues=[],
            voice_pass=True,
            tone_pass=True,
            revision_notes=notes,
        )

    def test_llm_judge_called_when_all_gates_pass(self, clean_draft, corpus, brief, mock_client):
        llm_result = self._make_llm_result(passed=True)
        with patch("app.validator.run_llm_judge", return_value=(llm_result, 300, 80, 0.01)) as mock_judge:
            verdict = validate_draft(clean_draft, corpus, brief, [], [], mock_client)
            mock_judge.assert_called_once()

        assert verdict.ship is True
        assert verdict.llm_judge is not None
        assert verdict.llm_judge.passed is True

    def test_ship_true_when_llm_passes(self, clean_draft, corpus, brief, mock_client):
        llm_result = self._make_llm_result(passed=True)
        with patch("app.validator.run_llm_judge", return_value=(llm_result, 300, 80, 0.01)):
            verdict = validate_draft(clean_draft, corpus, brief, [], [], mock_client)

        assert verdict.ship is True
        assert verdict.revision_notes is None
        assert verdict.cost_usd == pytest.approx(0.01)

    def test_ship_false_when_llm_fails(self, clean_draft, corpus, brief, mock_client):
        llm_result = self._make_llm_result(
            passed=False,
            notes="Type-D claim: 'some Type-D claim' has no source support.",
        )
        with patch("app.validator.run_llm_judge", return_value=(llm_result, 300, 80, 0.01)):
            verdict = validate_draft(clean_draft, corpus, brief, [], [], mock_client)

        assert verdict.ship is False
        assert verdict.revision_notes is not None
        assert "Type-D" in verdict.revision_notes

    def test_model_kwarg_passed_through(self, clean_draft, corpus, brief, mock_client):
        llm_result = self._make_llm_result(passed=True)
        with patch("app.validator.run_llm_judge", return_value=(llm_result, 300, 80, 0.01)) as mock_judge:
            validate_draft(
                clean_draft, corpus, brief, [], [], mock_client,
                model="claude-sonnet-4-6",
            )
            _, kwargs = mock_judge.call_args
            assert kwargs.get("model") == "claude-sonnet-4-6"

    def test_telemetry_recorded_on_llm_judge(self, clean_draft, corpus, brief, mock_client):
        llm_result = self._make_llm_result(passed=True)
        with patch("app.validator.run_llm_judge", return_value=(llm_result, 412, 95, 0.0134)):
            verdict = validate_draft(clean_draft, corpus, brief, [], [], mock_client)

        assert verdict.tokens_in == 412
        assert verdict.tokens_out == 95
        assert verdict.cost_usd == pytest.approx(0.0134)

    def test_certainty_inflation_fail_ship_false(self, clean_draft, corpus, brief, mock_client):
        llm_result = LLMJudgeResult(
            passed=False,
            grounding_pass=True,
            taxonomy_issues=[],
            certainty_inflation_issues=["BWS restores your energy within days."],
            voice_pass=True,
            tone_pass=True,
            revision_notes=(
                "Certainty inflation: 'restores your energy within days' — "
                "source says 'may support energy levels'. Soften to 'may support'."
            ),
        )
        with patch("app.validator.run_llm_judge", return_value=(llm_result, 300, 80, 0.01)):
            verdict = validate_draft(clean_draft, corpus, brief, [], [], mock_client)

        assert verdict.ship is False
        assert verdict.llm_judge.certainty_inflation_issues == ["BWS restores your energy within days."]
