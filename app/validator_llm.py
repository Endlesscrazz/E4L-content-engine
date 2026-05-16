"""
Opus 4.7 LLM judge for the Validator pipeline.

Runs only after all three deterministic gates pass (Layer 0/1/4 in validator_gates.py).
Uses tool use — not free-text parsing — for structured output. Two alternatives
were considered: JSON prefill (brittle, undocumented) and response parsing
(fragile when Opus adds prose). Tool use is the correct mechanism: typed contract,
loud failure if the model doesn't comply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.clients import AnthropicClient
from app.models import ContentBrief, Draft, LLMJudgeResult, SourceChunk

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "validator_judge.txt"

# t=0.0 — Validator must be consistent across identical drafts.
VALIDATOR_MODEL = "claude-opus-4-7"

# Tool schema — the judge MUST call this; no free-text response path exists.
# tool_choice={"type": "tool", "name": "submit_verdict"} enforces the call.
SUBMIT_VERDICT_TOOL: dict[str, Any] = {
    "name": "submit_verdict",
    "description": (
        "Submit the structured validation verdict after assessing the draft. "
        "Call this tool exactly once with all five check results populated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "grounding_pass": {
                "type": "boolean",
                "description": "True if every health/product claim traces to a cited source chunk.",
            },
            "taxonomy_issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact text of any Type-D claims (novel, not in source). "
                    "Empty list if none."
                ),
            },
            "certainty_inflation_issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact text of claims where asserted certainty exceeds "
                    "source-supported certainty."
                ),
            },
            "voice_pass": {
                "type": "boolean",
                "description": "True if draft follows Harry Massey's structural voice patterns.",
            },
            "tone_pass": {
                "type": "boolean",
                "description": "True if draft tone register matches the ContentBrief tone field.",
            },
            "passed": {
                "type": "boolean",
                "description": "True if and only if all five sub-checks pass.",
            },
            "revision_notes": {
                "type": "string",
                "description": (
                    "Required when passed=false. Specific, actionable notes for the Writer. "
                    "Quote the exact problematic phrase; name the check; suggest the fix direction."
                ),
            },
        },
        "required": [
            "grounding_pass",
            "taxonomy_issues",
            "certainty_inflation_issues",
            "voice_pass",
            "tone_pass",
            "passed",
        ],
    },
}


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _build_judge_message(
    draft: Draft,
    cited_chunks: list[SourceChunk],
    brief: ContentBrief,
    voice_anchors: list[SourceChunk],
) -> str:
    """Construct the user message for the Opus judge.

    Context discipline (architecture.md): Validator receives draft + cited chunks +
    voice anchors + brief axes only. Full corpus is never passed — keeps Opus
    context bounded and cost predictable per run.
    """
    parts: list[str] = []

    parts.append("## DRAFT TO VALIDATE")
    parts.append(f"Platform: {draft.platform.value}")
    if draft.subject:
        parts.append(f"Subject: {draft.subject}")
    parts.append(f"\n{draft.body}")

    parts.append("\n## CITED SOURCE CHUNKS")
    if cited_chunks:
        for chunk in cited_chunks:
            parts.append(f"\n[{chunk.chunk_id}] ({chunk.doc_name})")
            parts.append(chunk.content)
    else:
        parts.append("(no chunks cited — all claims must be Type-C general knowledge)")

    parts.append("\n## VOICE ANCHOR EXAMPLES (authentic Harry Massey voice)")
    if voice_anchors:
        for anchor in voice_anchors:
            parts.append(f"\n[{anchor.chunk_id}]")
            parts.append(anchor.content)
    else:
        parts.append("(no voice anchors provided — apply general voice criteria)")

    parts.append("\n## CONTENT BRIEF")
    parts.append(f"Goal: {brief.goal}")
    parts.append(f"Audience: {brief.audience.value}")
    parts.append(f"Funnel stage: {brief.funnel_stage.value}")
    parts.append(f"Tone: {brief.tone.value}")
    if brief.topic_focus:
        parts.append(f"Topic focus: {brief.topic_focus}")

    return "\n".join(parts)


def _coerce_list(val: Any) -> list[str]:
    """Guard against models returning a JSON-encoded string instead of a list.
    Sonnet occasionally stringifies array fields; Opus follows the schema correctly.
    This keeps _parse_verdict robust across both models in the comparison script."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val:
        import json
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return [val]
    return []


def _parse_verdict(tool_input: dict[str, Any]) -> LLMJudgeResult:
    """Map submit_verdict tool call arguments to LLMJudgeResult."""
    passed = tool_input["passed"]
    return LLMJudgeResult(
        passed=passed,
        grounding_pass=tool_input["grounding_pass"],
        taxonomy_issues=_coerce_list(tool_input.get("taxonomy_issues", [])),
        certainty_inflation_issues=_coerce_list(tool_input.get("certainty_inflation_issues", [])),
        voice_pass=tool_input["voice_pass"],
        tone_pass=tool_input["tone_pass"],
        revision_notes=tool_input.get("revision_notes") if not passed else None,
    )


def run_llm_judge(
    draft: Draft,
    cited_chunks: list[SourceChunk],
    brief: ContentBrief,
    voice_anchors: list[SourceChunk],
    client: AnthropicClient,
    model: str = VALIDATOR_MODEL,
) -> tuple[LLMJudgeResult, int, int, float]:
    """Run the LLM judge on a draft that has passed all deterministic gates.

    Returns (LLMJudgeResult, tokens_in, tokens_out, cost_usd).

    tool_choice forces submit_verdict — if the model returns text only or calls
    a different tool, we raise immediately. That's a model contract violation,
    not a recoverable draft failure.
    """
    system_prompt = _load_system_prompt()
    user_message = _build_judge_message(draft, cited_chunks, brief, voice_anchors)

    # temperature omitted — deprecated for Claude 4 models (Opus 4.7, Sonnet 4.6).
    # Tool use with a single forced tool already produces deterministic output.
    response, cost = client.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[SUBMIT_VERDICT_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        max_tokens=1024,
    )

    tool_block = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_verdict":
            tool_block = block
            break

    if tool_block is None:
        raise RuntimeError(
            f"Validator LLM judge did not call submit_verdict. "
            f"stop_reason={response.stop_reason}, "
            f"content_types={[b.type for b in response.content]}"
        )

    result = _parse_verdict(tool_block.input)
    return result, response.usage.input_tokens, response.usage.output_tokens, cost
