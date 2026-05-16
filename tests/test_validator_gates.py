"""
Deterministic validator gate tests — no LLM calls, no network, no DB.
All assertions use synthetic fixtures. Tests are authoritative for gate behavior.
"""

import pytest

from app.models import (
    Audience,
    Claim,
    ContentBrief,
    Draft,
    FunnelStage,
    Platform,
    SourceChunk,
)
from app.validator_gates import (
    DndFlag,
    check_citations,
    check_do_not_discuss,
    check_editorial,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _draft(body: str, subject: str | None = None, claims: list[Claim] | None = None) -> Draft:
    return Draft(
        platform=Platform.LINKEDIN,
        body=body,
        subject=subject,
        claims=claims or [],
    )


def _chunk(chunk_id: str, content: str, dnd_mode: str | None = None) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        doc_name="test_doc",
        content=content,
        doc_type="product",
        do_not_discuss=dnd_mode is not None,
        do_not_discuss_mode=dnd_mode,
    )


def _claim(text: str, chunk_id: str | None, substring: str | None) -> Claim:
    return Claim(text=text, chunk_id=chunk_id, cited_substring=substring)


def _brief(goal: str, topic_focus: str | None = None) -> ContentBrief:
    return ContentBrief(
        goal=goal,
        audience=Audience.CONSUMER,
        funnel_stage=FunnelStage.COLD,
        topic_focus=topic_focus,
    )


# ─── Layer 0: Editorial pre-gate ──────────────────────────────────────────────

class TestEditorialGate:

    # --- Hard-fail cases ---

    def test_cure_claim_fails(self):
        result = check_editorial(_draft("Infoceuticals cure chronic disease."))
        assert not result.passed
        assert result.triggered_phrase is not None
        assert "cure" in result.triggered_phrase.lower()

    def test_cures_variant_fails(self):
        result = check_editorial(_draft("The miHealth device cures fatigue."))
        assert not result.passed

    def test_diagnose_claim_fails(self):
        result = check_editorial(_draft("The scanner diagnoses your condition."))
        assert not result.passed
        assert "diagnose" in result.triggered_phrase.lower()

    def test_diagnosis_noun_fails(self):
        result = check_editorial(_draft("Get a real diagnosis with our technology."))
        assert not result.passed

    def test_treat_disease_fails(self):
        result = check_editorial(_draft("miHealth treats chronic illness effectively."))
        assert not result.passed
        assert "treat" in result.triggered_phrase.lower()

    def test_treat_disease_condition_fails(self):
        result = check_editorial(_draft("This device treats the condition at its source."))
        assert not result.passed

    def test_stop_medication_fails(self):
        result = check_editorial(_draft("Stop your medication and use miHealth instead."))
        assert not result.passed
        assert "medication" in result.triggered_phrase.lower()

    def test_discontinue_prescription_fails(self):
        result = check_editorial(_draft("You can discontinue your prescription entirely."))
        assert not result.passed

    def test_fda_approved_fails(self):
        result = check_editorial(_draft("This is an FDA-approved wellness device."))
        assert not result.passed
        assert "FDA" in result.triggered_phrase

    def test_absolute_efficacy_100pct_fails(self):
        result = check_editorial(_draft("Our technology is 100% effective for all users."))
        assert not result.passed
        assert "efficacy" in result.triggered_phrase.lower()

    def test_guaranteed_to_work_fails(self):
        result = check_editorial(_draft("Guaranteed to work for chronic pain."))
        assert not result.passed

    def test_anti_vaccine_fails(self):
        result = check_editorial(_draft("We support the anti-vaccine movement."))
        assert not result.passed
        assert "vaccine" in result.triggered_phrase.lower()

    def test_antivaccination_fails(self):
        result = check_editorial(_draft("Our approach is anti-vaccination."))
        assert not result.passed

    def test_permanently_cures_fails(self):
        # "permanently cures" hits the cure pattern first — either label is correct
        result = check_editorial(_draft("This permanently cures your fatigue."))
        assert not result.passed
        assert result.triggered_phrase is not None

    def test_reverses_chronic_disease_fails(self):
        result = check_editorial(_draft("The device reverses chronic disease progression."))
        assert not result.passed

    def test_email_subject_checked(self):
        # Violation in subject line, body is clean
        result = check_editorial(_draft(
            body="Learn about bioenergetic wellness.",
            subject="Cure your chronic fatigue today",
        ))
        assert not result.passed

    # --- Pass cases ---

    def test_supports_healing_passes(self):
        result = check_editorial(_draft("Supports your body's natural healing process."))
        assert result.passed

    def test_may_help_fatigue_passes(self):
        result = check_editorial(_draft("May help reduce fatigue symptoms over time."))
        assert result.passed

    def test_improve_energy_passes(self):
        result = check_editorial(_draft("Has been shown to improve energy levels in clients."))
        assert result.passed

    def test_treat_yourself_passes(self):
        # "treat yourself" is lifestyle copy — not a medical treatment claim
        result = check_editorial(_draft("Treat yourself to a better quality of life."))
        assert result.passed

    def test_treatment_plan_with_practitioner_passes(self):
        # "treatment plan" in practitioner context — not a device-diagnoses claim
        result = check_editorial(_draft(
            "Discuss a treatment plan with your licensed healthcare practitioner."
        ))
        assert result.passed

    def test_clean_email_passes(self):
        result = check_editorial(_draft(
            body="Discover how bioenergetic fields support your wellness journey.",
            subject="A new approach to energy and vitality",
        ))
        assert result.passed


# ─── Layer 1: Citation resolution ─────────────────────────────────────────────

class TestCitationResolutionGate:

    def test_valid_citation_passes(self):
        chunk = _chunk("c001", "The miHealth device uses NES field technology.")
        draft = _draft(
            body="miHealth uses NES field technology.",
            claims=[_claim("miHealth uses NES", "c001", "miHealth device uses NES field technology")],
        )
        result = check_citations(draft, {"c001": chunk})
        assert result.passed
        assert result.failed_claims == []

    def test_nonexistent_chunk_id_fails(self):
        draft = _draft(
            body="Some invented claim.",
            claims=[_claim("Invented claim", "does_not_exist", "invented substring")],
        )
        result = check_citations(draft, {})
        assert not result.passed
        assert "Invented claim" in result.failed_claims

    def test_substring_not_in_chunk_fails(self):
        chunk = _chunk("c001", "The device uses field resonance principles.")
        draft = _draft(
            body="Device uses quantum entanglement.",
            claims=[_claim("Quantum entanglement", "c001", "quantum entanglement mechanism")],
        )
        result = check_citations(draft, {"c001": chunk})
        assert not result.passed
        assert "Quantum entanglement" in result.failed_claims

    def test_type_c_claim_no_chunk_id_passes(self):
        # Type-C: general knowledge — chunk_id=None, no citation needed
        draft = _draft(
            body="Energy is fundamental to biological function.",
            claims=[_claim("Energy matters", chunk_id=None, substring=None)],
        )
        result = check_citations(draft, {})
        assert result.passed

    def test_claim_with_chunk_id_but_no_substring_skipped(self):
        # chunk_id present but cited_substring=None → skip (incomplete claim, not a violation)
        chunk = _chunk("c001", "Some content here.")
        draft = _draft(
            body="Some claim.",
            claims=[_claim("Some claim", chunk_id="c001", substring=None)],
        )
        result = check_citations(draft, {"c001": chunk})
        assert result.passed

    def test_smart_quote_normalization_passes(self):
        # Chunk stored with straight apostrophe; cited_substring uses docx smart apostrophe U+2019
        chunk = _chunk("c002", "Peter Fraser's research spanned 30 years.")
        draft = _draft(
            body="Research spanned 30 years.",
            claims=[_claim(
                "Fraser research",
                chunk_id="c002",
                substring="Peter Fraser’s research spanned 30 years",  # smart apostrophe
            )],
        )
        result = check_citations(draft, {"c002": chunk})
        assert result.passed

    def test_smart_double_quote_normalization_passes(self):
        chunk = _chunk("c003", 'He called it "bioenergetic wellness".')
        draft = _draft(
            body="Bioenergetic wellness defined.",
            claims=[_claim(
                "bioenergetic term",
                chunk_id="c003",
                substring="“bioenergetic wellness”",  # smart double quotes
            )],
        )
        result = check_citations(draft, {"c003": chunk})
        assert result.passed

    def test_partial_failure_reports_only_bad_claims(self):
        chunk = _chunk("c001", "NES field technology is foundational.")
        good = _claim("NES is foundational", "c001", "NES field technology is foundational")
        bad = _claim("Invented stat", "ghost_id", "completely made up")
        draft = _draft(body="NES is foundational. Invented stat.", claims=[good, bad])
        result = check_citations(draft, {"c001": chunk})
        assert not result.passed
        assert "Invented stat" in result.failed_claims
        assert "NES is foundational" not in result.failed_claims

    def test_empty_claims_list_passes(self):
        draft = _draft(body="Body with no structured claims.")
        result = check_citations(draft, {})
        assert result.passed

    def test_multiple_valid_claims_pass(self):
        chunks = {
            "c001": _chunk("c001", "miHealth improves cellular communication."),
            "c002": _chunk("c002", "BWS scans the body-field in 20 minutes."),
        }
        claims = [
            _claim("miHealth claim", "c001", "miHealth improves cellular communication"),
            _claim("BWS scan claim", "c002", "BWS scans the body-field in 20 minutes"),
        ]
        draft = _draft(body="miHealth improves cellular communication. BWS scans.", claims=claims)
        result = check_citations(draft, chunks)
        assert result.passed


# ─── Layer 4: do_not_discuss gate ─────────────────────────────────────────────

@pytest.fixture
def dnd_flags() -> list[DndFlag]:
    return [
        DndFlag(
            flag="peter_fraser_death",
            mode="never_in_generated_content",
            trigger_regex=(
                r"Peter Fraser.{0,60}(died|death|passed|deceased)"
                r"|passed away.{0,30}2012"
                r"|he finally ran into health problems he was unable to resolve"
            ),
            chunk_content_patterns=[
                "NOT to be discussed",
                "he finally ran into health problems he was unable to resolve",
            ],
        ),
        DndFlag(
            flag="tws_legacy_product",
            mode="do_not_volunteer",
            trigger_regex=r"Total Wellness System|Total WellNES|\bTWS\b|\bProVision\b",
            chunk_content_patterns=["Total Wellness System", "has not been sold since 2017"],
        ),
    ]


class TestDoNotDiscussGate:

    # --- never_in_generated_content ---

    def test_peter_fraser_body_text_fails(self, dnd_flags):
        # Parametric evasion: no citation, but draft text mentions Peter Fraser's death
        draft = _draft("Peter Fraser died in 2012 after health complications.")
        result = check_do_not_discuss(draft, [], dnd_flags)
        assert not result.passed
        assert result.triggered_flag == "peter_fraser_death"
        assert result.triggered_mode == "never_in_generated_content"

    def test_peter_fraser_alternate_phrase_fails(self, dnd_flags):
        draft = _draft("he finally ran into health problems he was unable to resolve")
        result = check_do_not_discuss(draft, [], dnd_flags)
        assert not result.passed
        assert result.triggered_flag == "peter_fraser_death"

    def test_peter_fraser_citation_join_fails(self, dnd_flags):
        # Draft cites the flagged chunk; body text is innocent
        flagged = _chunk(
            "energy4life_origin_story_0008",
            "NOT to be discussed: he finally ran into health problems he was unable to resolve",
            dnd_mode="never_in_generated_content",
        )
        draft = _draft(
            body="The founder dedicated his life to this technology.",
            claims=[],
        )
        draft.cited_chunk_ids = ["energy4life_origin_story_0008"]
        result = check_do_not_discuss(draft, [flagged], dnd_flags)
        assert not result.passed
        assert result.triggered_mode == "never_in_generated_content"

    def test_peter_fraser_email_subject_fails(self, dnd_flags):
        # Violation in email subject
        draft = _draft(
            body="Energy4Life has a rich history of innovation.",
            subject="Peter Fraser passed away in 2012 — his legacy continues",
        )
        result = check_do_not_discuss(draft, [], dnd_flags)
        assert not result.passed

    # --- do_not_volunteer ---

    def test_tws_without_brief_fails(self, dnd_flags):
        draft = _draft("The Total Wellness System was an early Energy4Life product.")
        result = check_do_not_discuss(draft, [], dnd_flags, brief=None)
        assert not result.passed
        assert result.triggered_flag == "tws_legacy_product"
        assert result.triggered_mode == "do_not_volunteer"

    def test_tws_brief_does_not_ask_fails(self, dnd_flags):
        draft = _draft("The Total Wellness System was an early product.")
        brief = _brief("Tell me about the miHealth device and its benefits.")
        result = check_do_not_discuss(draft, [], dnd_flags, brief=brief)
        assert not result.passed
        assert result.triggered_flag == "tws_legacy_product"

    def test_tws_brief_explicitly_asks_passes(self, dnd_flags):
        draft = _draft("The Total Wellness System (TWS) was an earlier scanning platform.")
        brief = _brief("Tell me about the TWS ProVision legacy system.")
        result = check_do_not_discuss(draft, [], dnd_flags, brief=brief)
        assert result.passed

    def test_provision_in_topic_focus_passes(self, dnd_flags):
        draft = _draft("The ProVision device was used before 2017.")
        brief = _brief("Overview of E4L products", topic_focus="ProVision legacy history")
        result = check_do_not_discuss(draft, [], dnd_flags, brief=brief)
        assert result.passed

    def test_tws_citation_join_no_brief_fails(self, dnd_flags):
        # Cited a do_not_volunteer chunk without user asking for it
        flagged = _chunk(
            "bws_chunk_001",
            "Total Wellness System has not been sold since 2017.",
            dnd_mode="do_not_volunteer",
        )
        draft = _draft("This technology has evolved significantly.", claims=[])
        draft.cited_chunk_ids = ["bws_chunk_001"]
        result = check_do_not_discuss(draft, [flagged], dnd_flags, brief=None)
        assert not result.passed
        assert result.triggered_mode == "do_not_volunteer"

    def test_tws_citation_join_with_brief_ask_passes(self, dnd_flags):
        flagged = _chunk(
            "bws_chunk_001",
            "Total Wellness System has not been sold since 2017.",
            dnd_mode="do_not_volunteer",
        )
        draft = _draft("The TWS was the predecessor to BWS.", claims=[])
        draft.cited_chunk_ids = ["bws_chunk_001"]
        brief = _brief("Explain the history of TWS ProVision.")
        result = check_do_not_discuss(draft, [flagged], dnd_flags, brief=brief)
        assert result.passed

    # --- Pass cases ---

    def test_clean_draft_passes(self, dnd_flags):
        draft = _draft("The miHealth device uses NES bioenergetic field technology.")
        result = check_do_not_discuss(draft, [], dnd_flags)
        assert result.passed

    def test_peter_fraser_name_only_passes(self, dnd_flags):
        # Name alone without death context should not trigger
        draft = _draft("Peter Fraser developed the NES Health system over three decades.")
        result = check_do_not_discuss(draft, [], dnd_flags)
        assert result.passed

    def test_clean_chunk_citation_passes(self, dnd_flags):
        clean = _chunk("chunk_normal", "The miHealth supports cellular wellness.")
        draft = _draft("miHealth supports wellness.", claims=[])
        draft.cited_chunk_ids = ["chunk_normal"]
        result = check_do_not_discuss(draft, [clean], dnd_flags)
        assert result.passed
