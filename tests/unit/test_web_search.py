import json

import httpx
import pytest
from pydantic import SecretStr

from ai_research_assistant.tools.web_search import (
    InvalidSearchResponseError,
    LazyTavilySearchService,
    SearchAuthenticationError,
    SearchConfigurationError,
    SearchRateLimitError,
    TavilySearchService,
)


def test_search_normalizes_and_deduplicates_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"

        payload = json.loads(request.content)
        assert payload["query"] == "LangGraph releases"
        assert payload["search_depth"] == "basic"
        assert payload["max_results"] == 5

        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "LangGraph documentation",
                        "url": "https://docs.langchain.com/oss/python/langgraph/overview",
                        "content": "Official LangGraph overview.",
                        "score": 0.95,
                    },
                    {
                        "title": "Duplicate documentation result",
                        "url": "https://docs.langchain.com/oss/python/langgraph/overview",
                        "content": "Duplicate content.",
                        "score": 0.80,
                    },
                    {
                        "title": "LangGraph repository",
                        "url": "https://github.com/langchain-ai/langgraph",
                        "content": "Official GitHub repository.",
                        "score": 0.90,
                    },
                    {
                        "title": "LangGraph releases",
                        "url": "https://github.com/langchain-ai/langgraph/releases",
                        "content": "Release history.",
                        "score": 0.85,
                    },
                ]
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = TavilySearchService(SecretStr("test-key"), post=client.post)
        results = service.search("  LangGraph releases  ")

    assert len(results) == 3
    assert [item.source.source_type for item in results] == [
        "documentation",
        "github",
        "release_notes",
    ]


def test_search_maps_authentication_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = TavilySearchService(SecretStr("invalid-key"), post=client.post)

        with pytest.raises(SearchAuthenticationError):
            service.search("LangGraph")


@pytest.mark.parametrize("status_code", [429, 432, 433])
def test_search_maps_usage_limits(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = TavilySearchService(SecretStr("test-key"), post=client.post)

        with pytest.raises(SearchRateLimitError):
            service.search("LangGraph")


def test_search_rejects_invalid_provider_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"title": "Broken result"}]},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = TavilySearchService(SecretStr("test-key"), post=client.post)

        with pytest.raises(InvalidSearchResponseError):
            service.search("LangGraph")


def test_search_rejects_empty_query() -> None:
    service = TavilySearchService(SecretStr("test-key"))

    with pytest.raises(ValueError, match="must not be empty"):
        service.search("   ")


def test_search_truncates_oversized_text_fields() -> None:
    long_title = "T" * 600
    long_content = "C" * 6000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": long_title,
                        "url": "https://docs.example.com/guide",
                        "content": long_content,
                        "score": 0.9,
                    }
                ]
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = TavilySearchService(
            SecretStr("test-key"),
            post=client.post,
        )
        results = service.search("Technical documentation")

    assert len(results[0].source.title) == 500
    assert len(results[0].content) == 5000


def test_lazy_search_creates_one_service_on_first_use() -> None:
    factory_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:

        def factory() -> TavilySearchService:
            nonlocal factory_calls
            factory_calls += 1
            return TavilySearchService(SecretStr("test-key"), post=client.post)

        service = LazyTavilySearchService(factory)

        assert factory_calls == 0
        assert service.search("LangGraph") == []
        assert service.search("LangChain") == []

    assert factory_calls == 1


def test_lazy_search_defers_configuration_error_until_first_use() -> None:
    def unconfigured_factory() -> TavilySearchService:
        raise SearchConfigurationError("WEB_SEARCH_API_KEY is not configured")

    service = LazyTavilySearchService(unconfigured_factory)

    with pytest.raises(SearchConfigurationError, match="WEB_SEARCH_API_KEY"):
        service.search("LangGraph")
