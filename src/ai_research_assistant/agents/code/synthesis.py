import json
from contextlib import suppress

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import ValidationError

from ai_research_assistant.agents.code.constants import INSUFFICIENT_CODE_RESPONSES
from ai_research_assistant.agents.code.evidence import _evidence_supports_request
from ai_research_assistant.agents.code.prompts import CODE_SYNTHESIS_SYSTEM_PROMPT
from ai_research_assistant.agents.code.types import CodeSynthesizer
from ai_research_assistant.agents.code.validation import (
    _attach_supporting_sources,
    _get_code_validation_feedback,
    _is_code_supported_by_evidence,
)
from ai_research_assistant.models import CodeCandidate, CodeResult, CodeSynthesis, EvidenceItem


def create_code_synthesizer(model: BaseChatModel) -> CodeSynthesizer:
    """Create a structured synthesis step that only receives collected evidence."""

    structured_model = model.with_structured_output(
        CodeCandidate,
        method="function_calling",
        include_raw=True,
    )

    def synthesize(
        query: str,
        response_language: str,
        evidence: list[EvidenceItem],
    ) -> CodeSynthesis:
        if not _evidence_supports_request(query, evidence):
            return _create_insufficient_synthesis()

        evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            indent=2,
        )

        messages = [
            ("system", CODE_SYNTHESIS_SYSTEM_PROMPT),
            (
                "human",
                (
                    f"User request:\n{query}\n\n"
                    f"Response language ISO code: {response_language}\n\n"
                    f"Evidence records:\n{evidence_json}"
                ),
            ),
        ]

        try:
            result = structured_model.invoke(messages)
        except (OutputParserException, ValidationError):
            return _create_insufficient_synthesis()

        candidate = _coerce_code_candidate(result)

        if candidate is None:
            return _create_insufficient_synthesis()

        synthesis = _attach_supporting_sources(
            query,
            _candidate_to_synthesis(candidate),
            evidence,
        )

        if _is_code_supported_by_evidence(query, synthesis, evidence):
            return synthesis

        validation_feedback = _get_code_validation_feedback(
            query,
            synthesis,
            evidence,
        )

        try:
            repaired_result = structured_model.invoke(
                [
                    *messages,
                    (
                        "human",
                        (
                            "The previous structured candidate failed deterministic "
                            "validation. Return one corrected replacement, not an "
                            "explanation of the error.\n\n"
                            f"Previous candidate:\n"
                            f"{candidate.model_dump_json(indent=2)}\n\n"
                            f"Validation feedback:\n- " + "\n- ".join(validation_feedback)
                        ),
                    ),
                ]
            )
        except (OutputParserException, ValidationError):
            return synthesis

        repaired_candidate = _coerce_code_candidate(repaired_result)

        if repaired_candidate is None:
            return synthesis

        return _attach_supporting_sources(
            query,
            _candidate_to_synthesis(repaired_candidate),
            evidence,
        )

    return synthesize


def _coerce_code_candidate(value: object) -> CodeCandidate | None:
    candidates: list[object] = [value]

    if isinstance(value, dict):
        candidates.append(value.get("parsed"))
        raw_message = value.get("raw")

        if isinstance(raw_message, AIMessage):
            candidates.extend(tool_call.get("args") for tool_call in raw_message.tool_calls)

        if isinstance(raw_message, BaseMessage) and isinstance(raw_message.content, str):
            with suppress(json.JSONDecodeError):
                candidates.append(json.loads(raw_message.content))

    for candidate in candidates:
        if isinstance(candidate, CodeCandidate):
            return candidate

        if not isinstance(candidate, dict):
            continue

        payload: object = candidate

        for wrapper_key in ("parameters", "arguments", "args"):
            wrapped_payload = candidate.get(wrapper_key)

            if isinstance(wrapped_payload, dict):
                payload = wrapped_payload
                break

        try:
            return CodeCandidate.model_validate(payload)
        except ValidationError:
            continue

    return None


def _candidate_to_synthesis(candidate: CodeCandidate) -> CodeSynthesis:
    return CodeSynthesis(
        explanation=candidate.explanation,
        code=candidate.code,
        code_language=candidate.code_language,
        used_source_ids=candidate.used_source_ids,
        is_sufficient=True,
    )


def _coerce_code_synthesis(value: object) -> CodeSynthesis:
    candidates: list[object] = [value]

    if isinstance(value, dict):
        candidates.append(value.get("parsed"))
        raw_message = value.get("raw")

        if isinstance(raw_message, AIMessage):
            candidates.extend(tool_call.get("args") for tool_call in raw_message.tool_calls)

        if isinstance(raw_message, BaseMessage) and isinstance(raw_message.content, str):
            with suppress(json.JSONDecodeError):
                candidates.append(json.loads(raw_message.content))

    for candidate in candidates:
        if isinstance(candidate, CodeSynthesis):
            return candidate

        if not isinstance(candidate, dict):
            continue

        payload: object = candidate

        for wrapper_key in ("parameters", "arguments", "args"):
            wrapped_payload = candidate.get(wrapper_key)

            if isinstance(wrapped_payload, dict):
                payload = wrapped_payload
                break

        try:
            return CodeSynthesis.model_validate(payload)
        except ValidationError:
            continue

    return _create_insufficient_synthesis()


def _create_insufficient_synthesis() -> CodeSynthesis:
    return CodeSynthesis(is_sufficient=False)


def _create_insufficient_result(language: str) -> CodeResult:
    resolved_language = language if language in INSUFFICIENT_CODE_RESPONSES else "en"
    return CodeResult(answer=INSUFFICIENT_CODE_RESPONSES[resolved_language])
