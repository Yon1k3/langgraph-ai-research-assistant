from uuid import uuid4

from langchain_core.messages import BaseMessage, ToolCall, ToolMessage
from pydantic import ValidationError

from ai_research_assistant.agents.research.constants import (
    INFORMATION_MARKERS,
    LIMITATION_INTENT_PATTERN,
    LIMITATION_SOURCE_HINT,
    MARKDOWN_IMAGE_PATTERN,
    MARKDOWN_LINK_PATTERN,
    MARKDOWN_TOKEN_PATTERN,
    MAX_SYNTHESIS_EVIDENCE_CONTENT_LENGTH,
    MAX_SYNTHESIS_EVIDENCE_ITEMS,
    OVERVIEW_SOURCE_HINT,
    RELEASE_INTENT_PATTERN,
    RELEASE_SOURCE_HINT,
    SENTENCE_SPLIT_PATTERN,
    SOURCE_TYPE_PRIORITY,
    SUBJECT_STOPWORDS,
    TECHNICAL_TERM_PATTERN,
    WHITESPACE_PATTERN,
)
from ai_research_assistant.errors import InvalidResearchResultError, ResearchSearchError
from ai_research_assistant.models import (
    EvidenceItem,
    SearchResultItem,
    SourceType,
    build_source_id,
)
from ai_research_assistant.tools import parse_search_service_error


def _create_initial_search_call(query: str) -> ToolCall:
    search_query = _build_initial_search_query(query)

    return ToolCall(
        name="search_web",
        args={
            "query": search_query,
            "max_results": 5,
        },
        id=f"initial-search-{uuid4()}",
        type="tool_call",
    )


def _build_initial_search_query(query: str) -> str:
    search_focus = query

    if not query.isascii():
        technical_terms: list[str] = []
        seen_terms: set[str] = set()

        for match in TECHNICAL_TERM_PATTERN.finditer(query):
            term = match.group(0).rstrip(".")

            if not term:
                continue

            term_key = term.casefold()

            if term_key in seen_terms:
                continue

            seen_terms.add(term_key)
            technical_terms.append(term)

        if technical_terms:
            search_focus = " ".join(technical_terms)

    if RELEASE_INTENT_PATTERN.search(query):
        source_hint = RELEASE_SOURCE_HINT
    elif LIMITATION_INTENT_PATTERN.search(query):
        source_hint = LIMITATION_SOURCE_HINT
    else:
        source_hint = OVERVIEW_SOURCE_HINT

    return f"{search_focus} {source_hint}"[:500]


def _create_evidence_item(item: SearchResultItem) -> EvidenceItem:
    source_url = str(item.source.url)

    return EvidenceItem(
        source_id=build_source_id(source_url),
        source=item.source,
        content=item.content,
        score=item.score,
    )


def _validate_initial_search_message(message: ToolMessage) -> None:
    if message.status == "error":
        if isinstance(message.content, str):
            detail = message.content
        else:
            detail = "Initial web search failed"

        search_error = parse_search_service_error(detail)

        if search_error is not None:
            raise search_error

        raise ResearchSearchError(detail)

    artifact = message.artifact

    if not isinstance(artifact, list) or not artifact:
        raise ResearchSearchError("Initial web search returned no verified sources")

    for raw_evidence in artifact:
        try:
            EvidenceItem.model_validate(raw_evidence)
        except ValidationError as exc:
            raise InvalidResearchResultError(
                "Initial search returned invalid evidence metadata"
            ) from exc


def _validate_messages(value: object) -> list[BaseMessage]:
    if not isinstance(value, list):
        raise InvalidResearchResultError("Research agent returned no message list")

    messages: list[BaseMessage] = []

    for item in value:
        if not isinstance(item, BaseMessage):
            raise InvalidResearchResultError("Research agent returned an invalid message")
        messages.append(item)

    return messages


def _extract_evidence(messages: list[BaseMessage]) -> list[EvidenceItem]:
    evidence_by_id: dict[str, EvidenceItem] = {}
    source_id_by_url: dict[str, str] = {}

    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != "search_web":
            continue

        artifact = message.artifact

        if artifact is None:
            continue

        if not isinstance(artifact, list):
            raise InvalidResearchResultError("Search tool returned an invalid evidence artifact")

        for raw_evidence in artifact:
            try:
                evidence = EvidenceItem.model_validate(raw_evidence)
            except ValidationError as exc:
                raise InvalidResearchResultError(
                    "Search tool returned invalid evidence metadata"
                ) from exc

            source_id = evidence.source_id
            url_key = str(evidence.source.url)
            existing_for_id = evidence_by_id.get(source_id)
            existing_id_for_url = source_id_by_url.get(url_key)

            if existing_for_id is not None and str(existing_for_id.source.url) != url_key:
                raise InvalidResearchResultError("Evidence source ID collision detected")

            if existing_id_for_url is not None and existing_id_for_url != source_id:
                raise InvalidResearchResultError("Evidence URL has conflicting source IDs")

            if existing_for_id is None or evidence.score > existing_for_id.score:
                evidence_by_id[source_id] = evidence
                source_id_by_url[url_key] = source_id

    return list(evidence_by_id.values())


def _rank_evidence(
    evidence: list[EvidenceItem],
    query: str,
) -> list[EvidenceItem]:
    source_type_priority = SOURCE_TYPE_PRIORITY

    if RELEASE_INTENT_PATTERN.search(query):
        source_type_priority = {
            "release_notes": 0,
            "documentation": 1,
            "github": 2,
            "web": 3,
        }

    return sorted(
        evidence,
        key=lambda item: (
            source_type_priority[item.source.source_type],
            -item.score,
            item.source_id,
        ),
    )


def _extract_subject_terms(query: str) -> list[str]:
    subject_terms: list[str] = []
    seen_terms: set[str] = set()

    for match in TECHNICAL_TERM_PATTERN.finditer(query):
        normalized_term = match.group(0).casefold().rstrip(".")

        if not normalized_term:
            continue

        if normalized_term in SUBJECT_STOPWORDS or normalized_term in seen_terms:
            continue

        seen_terms.add(normalized_term)
        subject_terms.append(normalized_term)

    return subject_terms


def _filter_relevant_evidence(
    evidence: list[EvidenceItem],
    query: str,
) -> list[EvidenceItem]:
    subject_terms = _extract_subject_terms(query)

    if not subject_terms:
        return evidence

    relevant_evidence = [
        item
        for item in evidence
        if any(
            term
            in " ".join(
                (
                    item.source.title,
                    str(item.source.url),
                    item.content,
                )
            ).casefold()
            for term in subject_terms
        )
    ]

    return relevant_evidence or evidence


def _select_relevant_excerpt(content: str, query: str, max_length: int) -> str:
    without_images = MARKDOWN_IMAGE_PATTERN.sub(" ", content)
    with_link_labels = MARKDOWN_LINK_PATTERN.sub(r"\1", without_images)
    without_markdown = MARKDOWN_TOKEN_PATTERN.sub(" ", with_link_labels)
    cleaned_content = WHITESPACE_PATTERN.sub(" ", without_markdown).strip()

    if not cleaned_content:
        cleaned_content = WHITESPACE_PATTERN.sub(" ", content).strip()

    if len(cleaned_content) <= max_length and content.count("](") < 3:
        return cleaned_content

    subject_terms = _extract_subject_terms(query)
    candidates: list[tuple[float, int, str]] = []

    for index, sentence in enumerate(SENTENCE_SPLIT_PATTERN.split(cleaned_content)):
        normalized_sentence = sentence.strip()

        if len(normalized_sentence) < 40:
            continue

        sentence_key = f" {normalized_sentence.casefold()} "
        subject_score = sum(sentence_key.count(term) for term in subject_terms) * 4
        information_score = sum(marker in sentence_key for marker in INFORMATION_MARKERS) * 2
        length_score = min(len(normalized_sentence), 300) / 300
        score = subject_score + information_score + length_score
        candidates.append((score, index, normalized_sentence))

    if not candidates:
        return _truncate_excerpt(cleaned_content, max_length)

    selected_candidates = sorted(candidates, reverse=True)[:10]
    selected_sentences = [
        sentence for _, _, sentence in sorted(selected_candidates, key=lambda item: item[1])
    ]
    excerpt = " ".join(selected_sentences)

    return _truncate_excerpt(excerpt, max_length)


def _truncate_excerpt(content: str, max_length: int) -> str:
    if len(content) <= max_length:
        return content

    truncated = content[:max_length]
    boundary_truncated = truncated.rsplit(" ", maxsplit=1)[0].strip()

    return boundary_truncated or truncated


def _prepare_synthesis_evidence(
    evidence: list[EvidenceItem],
    query: str,
) -> list[EvidenceItem]:
    relevant_evidence = _filter_relevant_evidence(evidence, query)
    ranked_evidence = _rank_evidence(relevant_evidence, query)
    authoritative_evidence: list[EvidenceItem] = []
    seen_source_types: set[SourceType] = set()

    for item in ranked_evidence:
        source_type = item.source.source_type

        if source_type == "web" or source_type in seen_source_types:
            continue

        seen_source_types.add(source_type)
        authoritative_evidence.append(item)

    selected_evidence = (authoritative_evidence or ranked_evidence)[:MAX_SYNTHESIS_EVIDENCE_ITEMS]

    return [
        item.model_copy(
            update={
                "content": _select_relevant_excerpt(
                    item.content,
                    query,
                    MAX_SYNTHESIS_EVIDENCE_CONTENT_LENGTH,
                ),
            }
        )
        for item in selected_evidence
    ]
