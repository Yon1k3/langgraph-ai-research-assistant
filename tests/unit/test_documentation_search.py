from collections.abc import Sequence

import pytest
from langchain_core.messages import ToolMessage

from ai_research_assistant.models import SearchResultItem, SourceItem
from ai_research_assistant.tools.documentation_search import (
    OfficialDocumentationSearchService,
    create_official_documentation_tool,
)


class FakeDomainSearchService:
    def __init__(self, results: list[SearchResultItem]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, list[str] | None]] = []

    def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_domains: Sequence[str] | None = None,
    ) -> list[SearchResultItem]:
        normalized_domains = list(include_domains) if include_domains is not None else None
        self.calls.append((query, max_results, normalized_domains))
        return self.results


def make_result(url: str, source_type: str = "web") -> SearchResultItem:
    return SearchResultItem(
        source=SourceItem(
            title="Documentation result",
            url=url,
            source_type=source_type,
        ),
        content="Official technical documentation content.",
        score=0.9,
    )


def test_documentation_search_enforces_domain_allowlist() -> None:
    provider = FakeDomainSearchService(
        [
            make_result("https://docs.langchain.com/oss/python/langgraph/overview"),
            make_result("https://blog.example.com/langgraph"),
        ]
    )
    service = OfficialDocumentationSearchService(provider)

    results = service.search(
        "LangGraph persistence",
        ["https://langchain.com/docs"],
        max_results=3,
    )

    assert provider.calls == [
        ("LangGraph persistence", 3, ["langchain.com"]),
    ]
    assert len(results) == 1
    assert results[0].source.url.host == "docs.langchain.com"
    assert results[0].source.source_type == "documentation"


def test_documentation_search_preserves_release_source_type() -> None:
    provider = FakeDomainSearchService(
        [make_result("https://github.com/example/project/releases", "release_notes")]
    )
    service = OfficialDocumentationSearchService(provider)

    results = service.search("Project releases", ["github.com"])

    assert results[0].source.source_type == "release_notes"


@pytest.mark.parametrize(
    "domains",
    [[], ["localhost"], ["*.example.com"], ["   "]],
)
def test_documentation_search_rejects_invalid_domains(domains: list[str]) -> None:
    service = OfficialDocumentationSearchService(FakeDomainSearchService([]))

    with pytest.raises(ValueError):
        service.search("Technical docs", domains)


def test_documentation_tool_returns_evidence_artifact() -> None:
    provider = FakeDomainSearchService(
        [make_result("https://docs.langchain.com/oss/python/langgraph/overview")]
    )
    tool = create_official_documentation_tool(OfficialDocumentationSearchService(provider))

    message = tool.invoke(
        {
            "name": "search_official_documentation",
            "args": {
                "query": "LangGraph overview",
                "domains": ["docs.langchain.com"],
            },
            "id": "documentation-call",
            "type": "tool_call",
        }
    )

    assert isinstance(message, ToolMessage)
    assert "docs.langchain.com" in str(message.content)
    assert isinstance(message.artifact, list)
    assert message.artifact[0]["source"]["source_type"] == "documentation"
