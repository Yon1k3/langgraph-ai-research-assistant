import re

from pydantic import ValidationError

from ai_research_assistant.agents.research.constants import (
    CYRILLIC_PATTERN,
    EXPERIENCE_ATTRIBUTION_PATTERN,
    EXPLICIT_LIMITATION_EVIDENCE_PATTERN,
    INSUFFICIENT_EVIDENCE_RESPONSES,
    LIMITATION_INTENT_PATTERN,
    LOW_LEVEL_ALTERNATIVE_PATTERN,
    PERSONAL_EXPERIENCE_PATTERN,
    SOURCE_HEADING_PATTERN,
    URL_LIST_ITEM_PATTERN,
    URL_PATTERN,
    WHITESPACE_PATTERN,
)
from ai_research_assistant.agents.research.evidence import _extract_subject_terms
from ai_research_assistant.errors import InvalidResearchResultError
from ai_research_assistant.models import EvidenceItem, GroundedClaim, SourceReference


def _select_verified_claims(
    claims: list[GroundedClaim],
    evidence: list[EvidenceItem],
    query: str,
) -> list[GroundedClaim]:
    evidence_by_id = {item.source_id: item for item in evidence}
    subject_terms = _extract_subject_terms(query)
    requires_limitation_evidence = bool(LIMITATION_INTENT_PATTERN.search(query))
    verified_claims: list[GroundedClaim] = []
    seen_statements: set[str] = set()

    for claim in claims:
        evidence_item = evidence_by_id.get(claim.source_id)

        if evidence_item is None:
            continue

        normalized_quote = _normalize_evidence_text(claim.supporting_quote)
        normalized_content = _normalize_evidence_text(evidence_item.content)

        if normalized_quote.casefold() not in normalized_content.casefold():
            continue

        if requires_limitation_evidence and not _has_explicit_limitation_evidence(normalized_quote):
            continue

        try:
            sanitized_statement = _sanitize_answer(claim.statement)
        except InvalidResearchResultError:
            continue

        if (
            evidence_item.source.source_type == "web"
            and PERSONAL_EXPERIENCE_PATTERN.search(normalized_quote)
            and not EXPERIENCE_ATTRIBUTION_PATTERN.search(sanitized_statement)
        ):
            sanitized_statement = _attribute_personal_experience(sanitized_statement)

        statement_key = sanitized_statement.casefold()

        if subject_terms and not any(term in statement_key for term in subject_terms):
            continue

        if statement_key in seen_statements:
            continue

        try:
            verified_claim = GroundedClaim.model_validate(
                {
                    **claim.model_dump(),
                    "statement": sanitized_statement,
                }
            )
        except ValidationError:
            continue

        seen_statements.add(statement_key)
        verified_claims.append(verified_claim)

    return verified_claims


def _has_explicit_limitation_evidence(quote: str) -> bool:
    return bool(
        EXPLICIT_LIMITATION_EVIDENCE_PATTERN.search(quote)
        or LOW_LEVEL_ALTERNATIVE_PATTERN.search(quote)
    )


def _attribute_personal_experience(statement: str) -> str:
    if CYRILLIC_PATTERN.search(statement):
        return f"Автор стороннього вебджерела повідомив, що {statement}"

    return f"The author of a third-party web source reported that {statement}"


def _normalize_evidence_text(content: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", content).strip()


def _format_verified_claims(claims: list[GroundedClaim]) -> str:
    return "\n\n".join(claim.statement for claim in claims)


def _resolve_used_sources(
    used_source_ids: list[str],
    evidence: list[EvidenceItem],
) -> list[SourceReference]:
    evidence_by_id = {item.source_id: item for item in evidence}
    unknown_source_ids = [
        source_id for source_id in used_source_ids if source_id not in evidence_by_id
    ]

    if unknown_source_ids:
        raise InvalidResearchResultError(
            "Research synthesizer referenced unknown evidence source IDs: "
            + ", ".join(unknown_source_ids)
        )

    return [
        SourceReference(
            source_id=source_id,
            source=evidence_by_id[source_id].source,
        )
        for source_id in used_source_ids
    ]


def _get_insufficient_evidence_response(response_language: str) -> str:
    return INSUFFICIENT_EVIDENCE_RESPONSES.get(
        response_language,
        INSUFFICIENT_EVIDENCE_RESPONSES["en"],
    )


def _sanitize_answer(answer: str) -> str:
    sanitized_lines: list[str] = []

    for line in answer.splitlines():
        if SOURCE_HEADING_PATTERN.fullmatch(line):
            continue

        if URL_PATTERN.search(line) and URL_LIST_ITEM_PATTERN.match(line):
            continue

        sanitized_lines.append(URL_PATTERN.sub("", line).rstrip())

    while sanitized_lines and not sanitized_lines[-1].strip():
        sanitized_lines.pop()

    sanitized = re.sub(r"\n{3,}", "\n\n", "\n".join(sanitized_lines)).strip()

    if not sanitized:
        raise InvalidResearchResultError(
            "Research agent returned no usable text after URL sanitization"
        )

    return sanitized
