from collections.abc import Sequence
from typing import Protocol
from urllib.parse import urlsplit

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from ai_research_assistant.models import EvidenceItem, SearchResultItem, build_source_id
from ai_research_assistant.tools.web_search import (
    SearchServiceError,
    create_lazy_tavily_search_service,
)

MAX_OFFICIAL_DOMAINS = 10
MAX_TOOL_CONTENT_LENGTH = 1_200


class DomainSearchService(Protocol):
    """Provider contract required by official documentation search."""

    def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_domains: Sequence[str] | None = None,
    ) -> list[SearchResultItem]:
        """Return search results, optionally restricted to domains."""


class OfficialDocumentationSearchService:
    """Search only explicitly trusted documentation domains."""

    def __init__(self, search_service: DomainSearchService) -> None:
        self._search_service = search_service

    def search(
        self,
        query: str,
        domains: Sequence[str],
        max_results: int = 5,
    ) -> list[SearchResultItem]:
        """Return results whose hosts match the trusted domain allowlist."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Documentation query must not be empty")

        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")

        normalized_domains = _normalize_domains(domains)
        results = self._search_service.search(
            normalized_query,
            max_results=max_results,
            include_domains=normalized_domains,
        )
        trusted_results: list[SearchResultItem] = []

        for item in results:
            host = (item.source.url.host or "").casefold().rstrip(".")

            if not any(_host_matches_domain(host, domain) for domain in normalized_domains):
                continue

            source = item.source

            if source.source_type not in {"github", "release_notes"}:
                source = source.model_copy(update={"source_type": "documentation"})

            trusted_results.append(item.model_copy(update={"source": source}))

        return trusted_results


class _OfficialDocumentationInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    domains: list[str] = Field(min_length=1, max_length=MAX_OFFICIAL_DOMAINS)
    max_results: int = Field(default=5, ge=1, le=10)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def create_official_documentation_search_service() -> OfficialDocumentationSearchService:
    """Create documentation search backed by the lazy Tavily adapter."""

    return OfficialDocumentationSearchService(create_lazy_tavily_search_service())


def create_official_documentation_tool(
    service: OfficialDocumentationSearchService,
) -> BaseTool:
    """Create a read-only LangChain tool for trusted documentation domains."""

    def search_official_documentation(
        query: str,
        domains: list[str],
        max_results: int = 5,
    ) -> tuple[str, list[dict[str, object]]]:
        """Search documentation only within verified official domains."""

        try:
            results = service.search(query, domains, max_results)
        except (SearchServiceError, ValueError) as exc:
            raise ToolException(str(exc)) from exc

        if not results:
            return "No official documentation results were found.", []

        return format_source_tool_results(results)

    return StructuredTool.from_function(
        func=search_official_documentation,
        name="search_official_documentation",
        description=(
            "Search technical documentation within an explicit allowlist of official "
            "domains. Use only domains already established as official sources."
        ),
        args_schema=_OfficialDocumentationInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
        handle_validation_error=(
            "Use a non-empty query, one to ten official domains, and max_results between 1 and 10."
        ),
    )


def _normalize_domains(domains: Sequence[str]) -> list[str]:
    if not domains:
        raise ValueError("At least one official documentation domain is required")

    if len(domains) > MAX_OFFICIAL_DOMAINS:
        raise ValueError(f"No more than {MAX_OFFICIAL_DOMAINS} domains are allowed")

    normalized_domains: list[str] = []

    for domain in domains:
        candidate = domain.strip()

        if not candidate or "*" in candidate:
            raise ValueError("Official documentation domains must be concrete host names")

        parsed = urlsplit(candidate if "://" in candidate else f"//{candidate}")
        host = (parsed.hostname or "").casefold().rstrip(".")

        if not host or "." not in host:
            raise ValueError("Official documentation domains must be valid host names")

        if host not in normalized_domains:
            normalized_domains.append(host)

    return normalized_domains


def _host_matches_domain(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def format_source_tool_results(
    results: Sequence[SearchResultItem],
) -> tuple[str, list[dict[str, object]]]:
    content_parts: list[str] = []
    artifacts: list[dict[str, object]] = []

    for item in results:
        source_id = build_source_id(str(item.source.url))
        evidence = EvidenceItem(
            source_id=source_id,
            source=item.source,
            content=item.content,
            score=item.score,
        )
        content_parts.append(
            "\n".join(
                [
                    f"[{source_id}] {item.source.title}",
                    f"URL: {item.source.url}",
                    f"Source type: {item.source.source_type}",
                    f"Content: {item.content[:MAX_TOOL_CONTENT_LENGTH]}",
                ]
            )
        )
        artifacts.append(evidence.model_dump(mode="json"))

    return "\n\n".join(content_parts), artifacts
