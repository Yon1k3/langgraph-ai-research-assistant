from collections.abc import Callable, Sequence
from typing import TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, SecretStr, ValidationError

from ai_research_assistant.config import get_settings
from ai_research_assistant.models import SearchResultItem, SourceItem, SourceType

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_SOURCE_TITLE_LENGTH = 500
MAX_SEARCH_CONTENT_LENGTH = 5000
SEARCH_TOOL_ERROR_PREFIX = "SEARCH_ERROR:"
PostCallable: TypeAlias = Callable[..., httpx.Response]


class SearchServiceError(RuntimeError):
    """Base error raised by the web search adapter."""


class SearchConfigurationError(SearchServiceError):
    """Raised when web search configuration is missing."""


class SearchAuthenticationError(SearchServiceError):
    """Raised when the search provider rejects the API key."""


class SearchRateLimitError(SearchServiceError):
    """Raised when the search provider usage limit is reached."""


class SearchUnavailableError(SearchServiceError):
    """Raised when the search provider cannot be reached."""


class InvalidSearchResponseError(SearchServiceError):
    """Raised when the search provider returns invalid data."""


class _TavilyResult(BaseModel):
    title: str = Field(min_length=1)
    url: HttpUrl
    content: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _TavilyResponse(BaseModel):
    results: list[_TavilyResult]

    model_config = ConfigDict(extra="ignore")


class TavilySearchService:
    """Search Tavily and normalize provider-specific responses."""

    def __init__(
        self,
        api_key: SecretStr,
        post: PostCallable = httpx.post,
    ) -> None:
        self._api_key = api_key
        self._post = post

    def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_domains: Sequence[str] | None = None,
    ) -> list[SearchResultItem]:
        """Return normalized, URL-deduplicated Tavily search results."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query must not be empty")

        if not 1 <= max_results <= 10:
            raise ValueError("max_results must be between 1 and 10")

        request_payload: dict[str, object] = {
            "query": normalized_query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        if include_domains:
            request_payload["include_domains"] = list(include_domains)

        try:
            response = self._post(
                TAVILY_SEARCH_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
                timeout=10.0,
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise SearchUnavailableError("Web search service is unavailable") from exc

        self._raise_for_status(response)

        try:
            payload = _TavilyResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise InvalidSearchResponseError("Web search service returned invalid data") from exc

        return self._normalize_results(payload.results)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == 401:
            raise SearchAuthenticationError("Web search API key was rejected")

        if response.status_code in {429, 432, 433}:
            raise SearchRateLimitError("Web search usage limit was reached")

        if response.is_error:
            raise SearchUnavailableError("Web search service returned an error")

    @staticmethod
    def _normalize_results(results: list[_TavilyResult]) -> list[SearchResultItem]:
        normalized: list[SearchResultItem] = []
        seen_urls: set[str] = set()

        for item in results:
            url_key = str(item.url)

            if url_key in seen_urls:
                continue

            seen_urls.add(url_key)
            normalized.append(
                SearchResultItem(
                    source=SourceItem(
                        title=item.title[:MAX_SOURCE_TITLE_LENGTH],
                        url=item.url,
                        source_type=_detect_source_type(item.url),
                    ),
                    content=item.content[:MAX_SEARCH_CONTENT_LENGTH],
                    score=item.score,
                )
            )

        return normalized


class LazyTavilySearchService:
    """Create the configured Tavily client only when search is first used."""

    def __init__(
        self,
        factory: Callable[[], TavilySearchService] | None = None,
    ) -> None:
        self._factory = factory or create_tavily_search_service
        self._service: TavilySearchService | None = None

    def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        include_domains: Sequence[str] | None = None,
    ) -> list[SearchResultItem]:
        """Delegate to one lazily created Tavily service instance."""

        if self._service is None:
            self._service = self._factory()

        return self._service.search(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
        )


def create_tavily_search_service() -> TavilySearchService:
    """Create the configured Tavily search service."""

    api_key = get_settings().web_search_api_key

    if api_key is None or not api_key.get_secret_value().strip():
        raise SearchConfigurationError("WEB_SEARCH_API_KEY is not configured")

    return TavilySearchService(api_key=api_key)


def create_lazy_tavily_search_service() -> LazyTavilySearchService:
    """Create a search adapter that defers settings validation until first use."""

    return LazyTavilySearchService()


def format_search_service_error(error: SearchServiceError) -> str:
    """Encode a safe provider-independent error category for a ToolMessage."""

    if isinstance(error, SearchConfigurationError):
        code = "configuration"
    elif isinstance(error, SearchAuthenticationError):
        code = "authentication"
    elif isinstance(error, SearchRateLimitError):
        code = "rate_limit"
    elif isinstance(error, InvalidSearchResponseError):
        code = "invalid_response"
    else:
        code = "unavailable"

    return f"{SEARCH_TOOL_ERROR_PREFIX}{code}"


def parse_search_service_error(message: str) -> SearchServiceError | None:
    """Restore a typed search failure from a safe ToolMessage marker."""

    error_types: dict[str, type[SearchServiceError]] = {
        "configuration": SearchConfigurationError,
        "authentication": SearchAuthenticationError,
        "rate_limit": SearchRateLimitError,
        "unavailable": SearchUnavailableError,
        "invalid_response": InvalidSearchResponseError,
    }

    if not message.startswith(SEARCH_TOOL_ERROR_PREFIX):
        return None

    code = message.removeprefix(SEARCH_TOOL_ERROR_PREFIX)
    error_type = error_types.get(code)

    if error_type is None:
        return None

    return error_type(f"Web search failed with category: {code}")


def _detect_source_type(url: HttpUrl) -> SourceType:
    host = (url.host or "").lower()
    path = (url.path or "").lower()

    if any(marker in path for marker in ("/releases", "/changelog", "/release-notes")):
        return "release_notes"

    if host == "github.com" or host.endswith(".github.com"):
        return "github"

    if (
        host.startswith("docs.")
        or host.endswith(".readthedocs.io")
        or "/docs/" in path
        or "/documentation/" in path
    ):
        return "documentation"

    return "web"
