import json

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import ValidationError

from ai_research_assistant.agents.research.prompts import SYNTHESIS_SYSTEM_PROMPT
from ai_research_assistant.agents.research.types import EvidenceSynthesizer
from ai_research_assistant.models import EvidenceItem, ResearchSynthesis


def create_grounded_synthesizer(model: BaseChatModel) -> EvidenceSynthesizer:
    """Create a structured synthesis step that only receives collected evidence."""

    structured_model = model.with_structured_output(
        ResearchSynthesis,
        method="function_calling",
        include_raw=True,
    )

    def synthesize(
        query: str,
        response_language: str,
        evidence: list[EvidenceItem],
    ) -> ResearchSynthesis:
        evidence_json = json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            indent=2,
        )
        try:
            result = structured_model.invoke(
                [
                    ("system", SYNTHESIS_SYSTEM_PROMPT),
                    (
                        "human",
                        (
                            f"User request:\n{query}\n\n"
                            f"Response language ISO code: {response_language}\n\n"
                            f"Evidence records:\n{evidence_json}"
                        ),
                    ),
                ]
            )
        except (OutputParserException, ValidationError):
            return _create_insufficient_synthesis()

        return _coerce_research_synthesis(result)

    return synthesize


def _coerce_research_synthesis(value: object) -> ResearchSynthesis:
    candidates: list[object] = [value]

    if isinstance(value, dict):
        candidates.append(value.get("parsed"))
        raw_message = value.get("raw")

        if isinstance(raw_message, AIMessage):
            candidates.extend(tool_call.get("args") for tool_call in raw_message.tool_calls)

        if isinstance(raw_message, BaseMessage) and isinstance(
            raw_message.content,
            str,
        ):
            candidates.extend(_extract_json_payloads(raw_message.content))

    for candidate in candidates:
        if isinstance(candidate, ResearchSynthesis):
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
            return ResearchSynthesis.model_validate(payload)
        except ValidationError:
            continue

    return _create_insufficient_synthesis()


def _extract_json_payloads(content: str) -> list[object]:
    payloads: list[object] = []

    for start_index, character in enumerate(content):
        if character != "{":
            continue

        json_object = _extract_balanced_json_object(content, start_index)

        if json_object is None:
            continue

        try:
            payloads.append(json.loads(json_object))
        except json.JSONDecodeError:
            continue

    return payloads


def _extract_balanced_json_object(content: str, start_index: int) -> str | None:
    depth = 0
    in_string = False
    is_escaped = False

    for index in range(start_index, len(content)):
        character = content[index]

        if in_string:
            if is_escaped:
                is_escaped = False
            elif character == "\\":
                is_escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1

            if depth == 0:
                return content[start_index : index + 1]

    return None


def _create_insufficient_synthesis() -> ResearchSynthesis:
    return ResearchSynthesis(
        claims=[],
        is_sufficient=False,
    )
