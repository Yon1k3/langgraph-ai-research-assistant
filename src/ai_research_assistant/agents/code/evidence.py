from urllib.parse import urlsplit
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from ai_research_assistant.agents.code.constants import (
    DOCUMENTATION_HOST_HINTS,
    GENERIC_REPOSITORY_TERMS,
    MAX_CODE_EVIDENCE_CONTENT_LENGTH,
    MAX_CODE_EVIDENCE_ITEMS,
    NON_DOCUMENTATION_HOSTS,
    REPOSITORY_STARS_PATTERN,
    TECHNICAL_TERM_PATTERN,
    URL_PATTERN,
)
from ai_research_assistant.agents.code.types import RepositoryCandidate
from ai_research_assistant.errors import CodeEvidenceError, InvalidCodeResultError
from ai_research_assistant.models import EvidenceItem, SourceReference


def _create_initial_repository_search_call(repository_query: str) -> ToolCall:
    return ToolCall(
        name="search_github_repositories",
        args={
            "query": repository_query,
            "max_results": 5,
        },
        id=f"code-source-{uuid4()}",
        type="tool_call",
    )


def _create_readme_call(owner: str, repository: str) -> ToolCall:
    return ToolCall(
        name="read_github_readme",
        args={
            "owner": owner,
            "repository": repository,
            "ref": None,
        },
        id=f"code-readme-{uuid4()}",
        type="tool_call",
    )


def _create_code_search_call(query: str, repository: str) -> ToolCall:
    return ToolCall(
        name="search_github_code",
        args={
            "query": query,
            "repository": repository,
            "max_results": 2,
        },
        id=f"code-search-{uuid4()}",
        type="tool_call",
    )


def _create_documentation_search_call(
    query: str,
    domains: list[str],
) -> ToolCall:
    search_terms = [
        *_build_repository_queries(query),
        *_get_required_code_tokens(query),
        "official example",
    ]
    return ToolCall(
        name="search_official_documentation",
        args={
            "query": " ".join(dict.fromkeys(search_terms)),
            "domains": domains,
            "max_results": 5,
        },
        id=f"code-documentation-{uuid4()}",
        type="tool_call",
    )


def _collect_initial_source_messages(
    query: str,
    repository_search_tool: BaseTool,
    documentation_tool: BaseTool | None,
    readme_tool: BaseTool | None,
    code_search_tool: BaseTool | None,
) -> list[BaseMessage]:
    repository_queries = _build_repository_queries(query)
    candidates: list[RepositoryCandidate] = []

    for repository_query in repository_queries:
        tool_call = _create_initial_repository_search_call(repository_query)
        search_message = repository_search_tool.invoke(tool_call)

        if not isinstance(search_message, ToolMessage):
            raise InvalidCodeResultError("Initial source search did not return a ToolMessage")

        for evidence in _extract_tool_message_evidence(search_message):
            candidates.append((repository_query, tool_call, search_message, evidence))

    selected = _select_repository_candidate(candidates)

    if selected is None:
        raise CodeEvidenceError("Initial GitHub source search returned no evidence")

    _, repository_call, _, repository_evidence = selected
    repository_name = repository_evidence.source.title
    repository_parts = repository_name.split("/", maxsplit=1)

    if len(repository_parts) != 2:
        raise InvalidCodeResultError("GitHub repository source has an invalid name")

    owner, repository = repository_parts
    selected_repository_message = ToolMessage(
        content=(
            f"[{repository_evidence.source_id}] {repository_evidence.source.title}\n"
            f"URL: {repository_evidence.source.url}\n"
            f"Content: {repository_evidence.content}"
        ),
        artifact=[repository_evidence.model_dump(mode="json")],
        tool_call_id=repository_call["id"],
        name=repository_call["name"],
    )
    messages: list[BaseMessage] = [
        HumanMessage(
            content=(
                f"Collect implementation evidence for this request:\n{query}\n\n"
                "The repository search below was filtered to the strongest exact-name "
                "candidate. Verify its README and relevant code before synthesis."
            )
        ),
        AIMessage(content="", tool_calls=[repository_call]),
        selected_repository_message,
    ]

    readme_evidence: list[EvidenceItem] = []

    if readme_tool is not None:
        readme_evidence = _append_successful_tool_call(
            messages,
            readme_tool,
            _create_readme_call(owner, repository),
        )

    documentation_domains = _extract_documentation_domains([repository_evidence, *readme_evidence])

    if documentation_tool is not None and documentation_domains:
        _append_successful_tool_call(
            messages,
            documentation_tool,
            _create_documentation_search_call(query, documentation_domains),
        )

    if code_search_tool is not None:
        for code_query in _build_code_search_queries(query, repository):
            _append_successful_tool_call(
                messages,
                code_search_tool,
                _create_code_search_call(code_query, repository_name),
            )

    messages.append(
        HumanMessage(
            content=(
                "Review the collected evidence. If it explicitly covers the requested "
                "API, finish evidence collection now without calling more tools. Call "
                "one additional tool only when an exact required API detail is missing."
            )
        )
    )

    return messages


def _extract_documentation_domains(evidence: list[EvidenceItem]) -> list[str]:
    preferred_hosts: list[str] = []
    fallback_hosts: list[str] = []

    for item in evidence:
        for url in URL_PATTERN.findall(item.content):
            host = (urlsplit(url.rstrip("),].")).hostname or "").casefold()

            if not host or host in NON_DOCUMENTATION_HOSTS:
                continue

            target = (
                preferred_hosts
                if any(hint in host for hint in DOCUMENTATION_HOST_HINTS)
                else fallback_hosts
            )

            if host not in target:
                target.append(host)

    return [*preferred_hosts, *fallback_hosts][:3]


def _append_successful_tool_call(
    messages: list[BaseMessage],
    tool: BaseTool,
    tool_call: ToolCall,
) -> list[EvidenceItem]:
    tool_message = tool.invoke(tool_call)

    if not isinstance(tool_message, ToolMessage):
        raise InvalidCodeResultError("Source tool did not return a ToolMessage")

    evidence = _extract_tool_message_evidence(tool_message)

    if not evidence:
        return []

    context_content = "\n\n".join(
        (f"[{item.source_id}] {item.source.title}\nURL: {item.source.url}\nContent: {item.content}")
        for item in evidence
    )
    context_message = ToolMessage(
        content=context_content,
        artifact=[item.model_dump(mode="json") for item in evidence],
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )

    messages.extend(
        [
            AIMessage(content="", tool_calls=[tool_call]),
            context_message,
        ]
    )

    return evidence


def _build_repository_queries(query: str) -> list[str]:
    terms: list[str] = []

    for match in TECHNICAL_TERM_PATTERN.findall(query):
        candidate = match.strip(".")
        normalized = candidate.casefold()

        if (
            not candidate
            or normalized in GENERIC_REPOSITORY_TERMS
            or (candidate.isupper() and len(candidate) > 1)
        ):
            continue

        if normalized not in {term.casefold() for term in terms}:
            terms.append(candidate)

        if len(terms) == 3:
            break

    if terms:
        return terms

    fallback_terms = TECHNICAL_TERM_PATTERN.findall(query)
    fallback_query = " ".join(fallback_terms[:3]) or query[:200]
    return [fallback_query]


def _build_code_search_queries(query: str, repository: str) -> list[str]:
    repository_name = repository.casefold()
    queries: list[str] = []

    for match in TECHNICAL_TERM_PATTERN.findall(query):
        candidate = match.strip(".")
        normalized = candidate.casefold()

        if (
            not candidate
            or normalized == repository_name
            or normalized in GENERIC_REPOSITORY_TERMS
            or (candidate.isupper() and len(candidate) > 1)
        ):
            continue

        if normalized not in {item.casefold() for item in queries}:
            queries.append(candidate)

        if len(queries) == 2:
            break

    for required_token in _get_required_code_tokens(query):
        if required_token not in queries:
            queries.append(required_token)

    return queries[:5]


def _get_required_code_tokens(query: str) -> list[str]:
    normalized_query = query.casefold()
    required_tokens: list[str] = []

    intent_tokens = (
        (("node", "вузол", "вузла", "вузли"), "add_node"),
        (("edge", "ребро", "ребра", "ребер"), "add_edge"),
        (("compile", "compilation", "компіляц", "скомпілю"), "compile"),
    )

    for markers, token in intent_tokens:
        if any(marker in normalized_query for marker in markers):
            required_tokens.append(token)

    for match in TECHNICAL_TERM_PATTERN.findall(query):
        candidate = match.strip(".")

        if candidate.isupper() and len(candidate) > 1:
            required_tokens.append(candidate)

    return list(dict.fromkeys(required_tokens))


def _extract_tool_message_evidence(message: ToolMessage) -> list[EvidenceItem]:
    if message.status == "error":
        return []

    if message.artifact is None:
        return []

    if not isinstance(message.artifact, list):
        raise InvalidCodeResultError("Source tool returned invalid evidence")

    evidence: list[EvidenceItem] = []

    for item in message.artifact:
        try:
            evidence.append(EvidenceItem.model_validate(item))
        except ValidationError as exc:
            raise InvalidCodeResultError("Source tool returned invalid evidence data") from exc

    return evidence


def _select_repository_candidate(
    candidates: list[RepositoryCandidate],
) -> RepositoryCandidate | None:
    if not candidates:
        return None

    exact_matches = [
        candidate
        for candidate in candidates
        if candidate[3].source.title.rsplit("/", maxsplit=1)[-1].casefold()
        == candidate[0].casefold()
    ]
    pool = exact_matches or candidates

    return max(
        pool,
        key=lambda candidate: (
            _get_repository_stars(candidate[3]),
            candidate[3].score,
        ),
    )


def _get_repository_stars(evidence: EvidenceItem) -> int:
    match = REPOSITORY_STARS_PATTERN.search(evidence.content)
    return int(match.group(1)) if match else 0


def _validate_messages(value: object) -> list[BaseMessage]:
    if not isinstance(value, list) or not all(
        isinstance(message, BaseMessage) for message in value
    ):
        raise InvalidCodeResultError("Code Agent returned invalid message state")

    return value


def _extract_evidence(messages: list[BaseMessage]) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []

    for message in messages:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue

        evidence.extend(_extract_tool_message_evidence(message))

    return evidence


def _prepare_evidence(
    evidence: list[EvidenceItem],
    query: str,
) -> list[EvidenceItem]:
    deduplicated: list[EvidenceItem] = []
    seen_urls: set[str] = set()

    for item in evidence:
        url = str(item.source.url)

        if url in seen_urls:
            continue

        seen_urls.add(url)
        deduplicated.append(item)

    search_terms = [
        *_build_repository_queries(query),
        *_get_required_code_tokens(query),
    ]
    ranked = sorted(
        deduplicated,
        key=lambda item: _get_evidence_rank(item, search_terms),
        reverse=True,
    )
    selected = _select_covering_evidence(ranked, search_terms)

    return [
        item.model_copy(
            update={
                "content": _select_relevant_content(item.content, search_terms),
            }
        )
        for item in selected
    ]


def _select_covering_evidence(
    ranked: list[EvidenceItem],
    search_terms: list[str],
) -> list[EvidenceItem]:
    selected: list[EvidenceItem] = []
    remaining_terms = {term.casefold() for term in search_terms if term}

    while remaining_terms and len(selected) < MAX_CODE_EVIDENCE_ITEMS:
        candidates = [item for item in ranked if item not in selected]

        if not candidates:
            break

        best = max(
            candidates,
            key=lambda item: len(_get_matching_terms(item.content, remaining_terms)),
        )
        covered_terms = _get_matching_terms(best.content, remaining_terms)

        if not covered_terms:
            break

        selected.append(best)
        remaining_terms -= covered_terms

    for item in ranked:
        if len(selected) == MAX_CODE_EVIDENCE_ITEMS:
            break

        if item not in selected:
            selected.append(item)

    return selected


def _get_matching_terms(content: str, terms: set[str]) -> set[str]:
    normalized_content = content.replace("\\_", "_").casefold()
    return {term for term in terms if term in normalized_content}


def _get_evidence_rank(
    evidence: EvidenceItem,
    search_terms: list[str],
) -> tuple[int, int, float]:
    normalized_content = evidence.content.replace("\\_", "_").casefold()
    required_terms = {term.casefold() for term in search_terms if "_" in term}
    exact_matches = sum(normalized_content.count(term.casefold()) for term in search_terms if term)
    required_matches = sum(normalized_content.count(term) for term in required_terms)
    source_priority = {
        "documentation": 3,
        "github": 2,
        "release_notes": 1,
        "web": 0,
    }[evidence.source.source_type]

    return (
        required_matches * 10 + exact_matches,
        source_priority,
        evidence.score,
    )


def _select_relevant_content(content: str, search_terms: list[str]) -> str:
    normalized_content = content.replace("\\_", "_")

    if len(normalized_content) <= MAX_CODE_EVIDENCE_CONTENT_LENGTH:
        return normalized_content

    casefolded_content = normalized_content.casefold()
    positions = sorted(
        {
            casefolded_content.find(term.casefold())
            for term in search_terms
            if term and casefolded_content.find(term.casefold()) >= 0
        }
    )

    if not positions:
        return normalized_content[:MAX_CODE_EVIDENCE_CONTENT_LENGTH]

    excerpts: list[str] = []
    remaining_length = MAX_CODE_EVIDENCE_CONTENT_LENGTH

    for position in positions:
        if remaining_length <= 0:
            break

        start = max(position - 250, 0)
        excerpt_length = min(700, remaining_length)
        excerpt = normalized_content[start : start + excerpt_length]

        if excerpt not in excerpts:
            excerpts.append(excerpt)
            remaining_length -= len(excerpt)

    return "\n...\n".join(excerpts)[:MAX_CODE_EVIDENCE_CONTENT_LENGTH]


def _resolve_used_sources(
    source_ids: list[str],
    evidence: list[EvidenceItem],
) -> list[SourceReference]:
    evidence_by_id = {item.source_id: item for item in evidence}
    unique_source_ids = list(dict.fromkeys(source_ids))

    if any(source_id not in evidence_by_id for source_id in unique_source_ids):
        return []

    return [
        SourceReference(
            source_id=evidence_by_id[source_id].source_id,
            source=evidence_by_id[source_id].source,
        )
        for source_id in unique_source_ids
    ]


def _evidence_supports_request(
    query: str,
    evidence: list[EvidenceItem],
) -> bool:
    normalized_evidence = (
        "\n".join(item.content for item in evidence).replace("\\_", "_").casefold()
    )
    expected_tokens = [
        *_build_repository_queries(query),
        *_get_required_code_tokens(query),
    ]

    return bool(expected_tokens) and all(
        token.casefold() in normalized_evidence for token in expected_tokens
    )
