import httpx
import pytest
from pydantic import SecretStr

from ai_research_assistant.tools.github_search import (
    GITHUB_API_VERSION,
    GitHubAuthenticationError,
    GitHubConfigurationError,
    GitHubInvalidRequestError,
    GitHubNotFoundError,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubSearchService,
    GitHubUnavailableError,
    InvalidGitHubResponseError,
    create_github_tools,
)


def repository_payload() -> dict[str, object]:
    return {
        "items": [
            {
                "full_name": "langchain-ai/langgraph",
                "html_url": "https://github.com/langchain-ai/langgraph",
                "description": "Build resilient language agents as graphs.",
                "stargazers_count": 25000,
                "language": "Python",
                "topics": ["agents", "langgraph"],
                "default_branch": "main",
                "homepage": "https://docs.langchain.com/oss/python/langgraph/overview",
                "updated_at": "2026-07-01T10:00:00Z",
                "score": 1.2,
            }
        ]
    }


def test_repository_search_normalizes_metadata_and_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-GitHub-Api-Version"] == GITHUB_API_VERSION
        assert request.headers["User-Agent"] == "langgraph-ai-research-assistant"
        assert "Authorization" not in request.headers
        assert request.url.params["q"] == "LangGraph in:name is:public"
        assert request.url.params["sort"] == "stars"
        return httpx.Response(200, json=repository_payload(), request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(get=client.get)
        results = service.search_repositories("  LangGraph in:name  ", max_results=3)

    assert len(results) == 1
    assert results[0].source.title == "langchain-ai/langgraph"
    assert results[0].source.source_type == "github"
    assert results[0].score == 1.0
    assert "Stars: 25000" in results[0].content
    assert "docs.langchain.com" in results[0].content


def test_readme_uses_raw_media_type_and_cache() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.headers["Accept"] == "application/vnd.github.raw+json"
        assert request.url.params["ref"] == "main"
        return httpx.Response(
            200,
            text="# LangGraph\nOfficial repository README.",
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(get=client.get)
        first = service.get_readme("langchain-ai", "langgraph", "main")
        first.content = "mutated by caller"
        second = service.get_readme("langchain-ai", "langgraph", "main")

    assert request_count == 1
    assert second.content.startswith("# LangGraph")
    assert str(second.source.url).endswith("langchain-ai/langgraph#readme")


def test_repository_cache_expires() -> None:
    request_count = 0
    now = [100.0]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json=repository_payload(), request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(
            get=client.get,
            cache_ttl_seconds=10,
            clock=lambda: now[0],
        )
        service.search_repositories("LangGraph")
        service.search_repositories("LangGraph")
        now[0] = 111.0
        service.search_repositories("LangGraph")

    assert request_count == 2


def test_release_search_normalizes_release_notes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/langchain-ai/langgraph/releases")
        return httpx.Response(
            200,
            json=[
                {
                    "name": "LangGraph 2.0",
                    "tag_name": "2.0.0",
                    "html_url": "https://github.com/langchain-ai/langgraph/releases/tag/2.0.0",
                    "body": "Adds a new durable execution API.",
                    "published_at": "2026-06-01T10:00:00Z",
                    "prerelease": False,
                    "draft": False,
                }
            ],
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(get=client.get)
        results = service.list_releases("langchain-ai", "langgraph")

    assert results[0].source.source_type == "release_notes"
    assert "Tag: 2.0.0" in results[0].content
    assert "durable execution" in results[0].content


def test_code_search_requires_token_before_request() -> None:
    service = GitHubSearchService()

    with pytest.raises(GitHubConfigurationError, match="GITHUB_TOKEN"):
        service.search_code("StateGraph")


def test_code_search_filters_private_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "name": "secret.py",
                        "path": "secret.py",
                        "html_url": "https://github.com/private/example/blob/main/secret.py",
                        "repository": {
                            "full_name": "private/example",
                            "private": True,
                        },
                    }
                ]
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(SecretStr("test-token"), get=client.get)

        assert service.search_code("secret") == []


def test_code_search_uses_token_repository_qualifier_and_text_matches() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.headers["Accept"] == "application/vnd.github.text-match+json"
        assert request.url.params["q"] == "StateGraph repo:langchain-ai/langgraph"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "name": "state.py",
                        "path": "libs/langgraph/langgraph/graph/state.py",
                        "html_url": (
                            "https://github.com/langchain-ai/langgraph/blob/main/"
                            "libs/langgraph/langgraph/graph/state.py"
                        ),
                        "repository": {"full_name": "langchain-ai/langgraph"},
                        "text_matches": [{"fragment": "class StateGraph(Generic[StateT]):"}],
                    }
                ]
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(SecretStr("test-token"), get=client.get)
        results = service.search_code(
            "StateGraph",
            repository="langchain-ai/langgraph",
        )

    assert len(results) == 1
    assert "class StateGraph" in results[0].content
    assert results[0].source.source_type == "github"


@pytest.mark.parametrize(
    ("status_code", "headers", "error_type"),
    [
        (401, {}, GitHubAuthenticationError),
        (403, {}, GitHubPermissionError),
        (403, {"x-ratelimit-remaining": "0"}, GitHubRateLimitError),
        (404, {}, GitHubNotFoundError),
        (422, {}, GitHubInvalidRequestError),
        (429, {}, GitHubRateLimitError),
        (500, {}, GitHubUnavailableError),
    ],
)
def test_github_status_errors_are_typed(
    status_code: int,
    headers: dict[str, str],
    error_type: type[Exception],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers=headers,
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(get=client.get)

        with pytest.raises(error_type):
            service.search_repositories("LangGraph")


def test_github_rejects_invalid_response_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"items": [{"broken": "repository"}]},
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        service = GitHubSearchService(get=client.get)

        with pytest.raises(InvalidGitHubResponseError):
            service.search_repositories("LangGraph")


def test_github_tool_factory_exposes_only_read_operations() -> None:
    tools = create_github_tools(GitHubSearchService())

    assert [tool.name for tool in tools] == [
        "search_github_repositories",
        "read_github_readme",
        "list_github_releases",
        "search_github_code",
    ]
