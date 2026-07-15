import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeAlias, TypeVar

import httpx
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    TypeAdapter,
    ValidationError,
)

from ai_research_assistant.config import get_settings
from ai_research_assistant.models import SearchResultItem, SourceItem
from ai_research_assistant.tools.documentation_search import format_source_tool_results

GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"
GITHUB_USER_AGENT = "langgraph-ai-research-assistant"
MAX_GITHUB_RESULTS = 10
MAX_GITHUB_CONTENT_LENGTH = 5_000
DEFAULT_CACHE_TTL_SECONDS = 300.0
MAX_CACHE_ENTRIES = 64
REPOSITORY_PART_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
GetCallable: TypeAlias = Callable[..., httpx.Response]
Clock: TypeAlias = Callable[[], float]
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)


class GitHubServiceError(RuntimeError):
    """Base error raised by the GitHub read-only adapter."""


class GitHubConfigurationError(GitHubServiceError):
    """Raised when an operation requires missing configuration."""


class GitHubAuthenticationError(GitHubServiceError):
    """Raised when GitHub rejects the configured token."""


class GitHubPermissionError(GitHubServiceError):
    """Raised when a token cannot access the requested resource."""


class GitHubRateLimitError(GitHubServiceError):
    """Raised when a GitHub API rate limit is exhausted."""


class GitHubNotFoundError(GitHubServiceError):
    """Raised when a requested repository resource does not exist."""


class GitHubInvalidRequestError(GitHubServiceError):
    """Raised when GitHub rejects request parameters."""


class GitHubUnavailableError(GitHubServiceError):
    """Raised when GitHub cannot be reached or returns a server error."""


class InvalidGitHubResponseError(GitHubServiceError):
    """Raised when GitHub returns malformed or unsupported data."""


class _GitHubRepository(BaseModel):
    full_name: str = Field(min_length=1)
    html_url: HttpUrl
    description: str | None = None
    stargazers_count: int = Field(default=0, ge=0)
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    default_branch: str = Field(default="main", min_length=1)
    homepage: str | None = None
    updated_at: str | None = None
    score: float | None = None
    private: bool = False

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _RepositorySearchResponse(BaseModel):
    items: list[_GitHubRepository]

    model_config = ConfigDict(extra="ignore")


class _GitHubRelease(BaseModel):
    name: str | None = None
    tag_name: str = Field(min_length=1)
    html_url: HttpUrl
    body: str | None = None
    published_at: str | None = None
    prerelease: bool = False
    draft: bool = False

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _GitHubCodeRepository(BaseModel):
    full_name: str = Field(min_length=1)
    private: bool = False

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _GitHubTextMatch(BaseModel):
    fragment: str = Field(min_length=1)

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _GitHubCodeItem(BaseModel):
    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    html_url: HttpUrl
    repository: _GitHubCodeRepository
    text_matches: list[_GitHubTextMatch] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _CodeSearchResponse(BaseModel):
    items: list[_GitHubCodeItem]

    model_config = ConfigDict(extra="ignore")


@dataclass(frozen=True)
class _CacheEntry(Generic[T]):
    expires_at: float
    value: T


class _TTLCache(Generic[T]):
    """Small process-local cache that never hides data indefinitely."""

    def __init__(
        self,
        ttl_seconds: float,
        clock: Clock,
        max_entries: int = MAX_CACHE_ENTRIES,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("cache TTL must not be negative")

        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._max_entries = max_entries
        self._entries: dict[tuple[str, ...], _CacheEntry[T]] = {}

    def get(self, key: tuple[str, ...]) -> T | None:
        entry = self._entries.get(key)

        if entry is None:
            return None

        if entry.expires_at <= self._clock():
            del self._entries[key]
            return None

        return entry.value

    def set(self, key: tuple[str, ...], value: T) -> None:
        if self._ttl_seconds == 0:
            return

        if key not in self._entries and len(self._entries) >= self._max_entries:
            oldest_key = next(iter(self._entries))
            del self._entries[oldest_key]

        self._entries[key] = _CacheEntry(
            expires_at=self._clock() + self._ttl_seconds,
            value=value,
        )


class GitHubSearchService:
    """Read public GitHub metadata and optionally authenticated code search."""

    def __init__(
        self,
        token: SecretStr | None = None,
        get: GetCallable = httpx.get,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        clock: Clock = monotonic,
    ) -> None:
        self._token = _normalize_token(token)
        self._get = get
        self._repository_cache = _TTLCache[tuple[SearchResultItem, ...]](
            cache_ttl_seconds,
            clock,
        )
        self._readme_cache = _TTLCache[SearchResultItem](cache_ttl_seconds, clock)
        self._release_cache = _TTLCache[tuple[SearchResultItem, ...]](
            cache_ttl_seconds,
            clock,
        )
        self._code_cache = _TTLCache[tuple[SearchResultItem, ...]](
            cache_ttl_seconds,
            clock,
        )

    def search_repositories(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResultItem]:
        """Search public repository metadata without requiring a token."""

        normalized_query = _normalize_query(query, "Repository search query")
        _validate_max_results(max_results)
        cache_key = (normalized_query, str(max_results))
        cached = self._repository_cache.get(cache_key)

        if cached is not None:
            return _copy_results(cached)

        response = self._request(
            f"{GITHUB_API_URL}/search/repositories",
            params={
                "q": f"{normalized_query} is:public",
                "per_page": max_results,
                "sort": "stars",
                "order": "desc",
            },
        )
        payload = _validate_json_response(response, _RepositorySearchResponse)
        results = tuple(
            _repository_to_search_result(item, index)
            for index, item in enumerate(payload.items)
            if not item.private
        )
        self._repository_cache.set(cache_key, results)
        return _copy_results(results)

    def get_readme(
        self,
        owner: str,
        repository: str,
        ref: str | None = None,
    ) -> SearchResultItem:
        """Read a public repository README as raw text."""

        normalized_owner = _validate_repository_part(owner, "owner")
        normalized_repository = _validate_repository_part(repository, "repository")
        normalized_ref = ref.strip() if ref is not None and ref.strip() else ""
        cache_key = (normalized_owner, normalized_repository, normalized_ref)
        cached = self._readme_cache.get(cache_key)

        if cached is not None:
            return cached.model_copy(deep=True)

        params: dict[str, object] | None = {"ref": normalized_ref} if normalized_ref else None
        response = self._request(
            f"{GITHUB_API_URL}/repos/{normalized_owner}/{normalized_repository}/readme",
            params=params,
            accept="application/vnd.github.raw+json",
        )
        content = response.text.strip()

        if not content:
            raise InvalidGitHubResponseError("GitHub returned an empty README")

        result = SearchResultItem(
            source=SourceItem(
                title=f"{normalized_owner}/{normalized_repository} README",
                url=HttpUrl(
                    f"https://github.com/{normalized_owner}/{normalized_repository}#readme"
                ),
                source_type="github",
            ),
            content=content[:MAX_GITHUB_CONTENT_LENGTH],
            score=1.0,
        )
        self._readme_cache.set(cache_key, result)
        return result.model_copy(deep=True)

    def list_releases(
        self,
        owner: str,
        repository: str,
        max_results: int = 5,
    ) -> list[SearchResultItem]:
        """Return published GitHub releases for a public repository."""

        normalized_owner = _validate_repository_part(owner, "owner")
        normalized_repository = _validate_repository_part(repository, "repository")
        _validate_max_results(max_results)
        cache_key = (normalized_owner, normalized_repository, str(max_results))
        cached = self._release_cache.get(cache_key)

        if cached is not None:
            return _copy_results(cached)

        response = self._request(
            f"{GITHUB_API_URL}/repos/{normalized_owner}/{normalized_repository}/releases",
            params={"per_page": max_results},
        )

        try:
            releases = TypeAdapter(list[_GitHubRelease]).validate_python(response.json())
        except (ValueError, ValidationError) as exc:
            raise InvalidGitHubResponseError("GitHub returned invalid release data") from exc

        results = tuple(
            _release_to_search_result(release, index)
            for index, release in enumerate(releases)
            if not release.draft
        )
        self._release_cache.set(cache_key, results)
        return _copy_results(results)

    def search_code(
        self,
        query: str,
        repository: str | None = None,
        max_results: int = 5,
    ) -> list[SearchResultItem]:
        """Search GitHub code; GitHub requires authentication for this endpoint."""

        if self._token is None:
            raise GitHubConfigurationError("GITHUB_TOKEN is required for GitHub code search")

        normalized_query = _normalize_query(query, "Code search query")
        _validate_max_results(max_results)
        normalized_repository = ""

        if repository is not None and repository.strip():
            normalized_repository = _validate_repository_name(repository)

        qualified_query = normalized_query

        if normalized_repository:
            qualified_query = f"{normalized_query} repo:{normalized_repository}"

        cache_key = (qualified_query, str(max_results))
        cached = self._code_cache.get(cache_key)

        if cached is not None:
            return _copy_results(cached)

        response = self._request(
            f"{GITHUB_API_URL}/search/code",
            params={"q": qualified_query, "per_page": max_results},
            accept="application/vnd.github.text-match+json",
        )
        payload = _validate_json_response(response, _CodeSearchResponse)
        results = tuple(
            _code_item_to_search_result(item, index)
            for index, item in enumerate(payload.items)
            if not item.repository.private
        )
        self._code_cache.set(cache_key, results)
        return _copy_results(results)

    def _request(
        self,
        url: str,
        *,
        params: dict[str, object] | None = None,
        accept: str = "application/vnd.github+json",
    ) -> httpx.Response:
        headers = {
            "Accept": accept,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": GITHUB_USER_AGENT,
        }

        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token.get_secret_value()}"

        try:
            response = self._get(
                url,
                headers=headers,
                params=params,
                timeout=10.0,
                follow_redirects=True,
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            raise GitHubUnavailableError("GitHub API is unavailable") from exc

        _raise_for_status(response)
        return response


class _RepositorySearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=MAX_GITHUB_RESULTS)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _ReadmeInput(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    repository: str = Field(min_length=1, max_length=100)
    ref: str | None = Field(default=None, max_length=200)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _ReleaseInput(BaseModel):
    owner: str = Field(min_length=1, max_length=100)
    repository: str = Field(min_length=1, max_length=100)
    max_results: int = Field(default=5, ge=1, le=MAX_GITHUB_RESULTS)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class _CodeSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    repository: str | None = Field(default=None, max_length=201)
    max_results: int = Field(default=5, ge=1, le=MAX_GITHUB_RESULTS)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def create_github_search_service() -> GitHubSearchService:
    """Create the GitHub adapter with an optional read-only token."""

    return GitHubSearchService(token=get_settings().github_token)


def create_github_tools(service: GitHubSearchService) -> list[BaseTool]:
    """Create read-only LangChain tools backed by one shared GitHub service."""

    def search_github_repositories(
        query: str,
        max_results: int = 5,
    ) -> tuple[str, list[dict[str, object]]]:
        """Find public GitHub repositories and return normalized metadata."""

        return _run_result_tool(service.search_repositories, query, max_results)

    def read_github_readme(
        owner: str,
        repository: str,
        ref: str | None = None,
    ) -> tuple[str, list[dict[str, object]]]:
        """Read a public repository README without modifying the repository."""

        try:
            result = service.get_readme(owner, repository, ref)
        except (GitHubServiceError, ValueError) as exc:
            raise ToolException(str(exc)) from exc

        return format_source_tool_results([result])

    def list_github_releases(
        owner: str,
        repository: str,
        max_results: int = 5,
    ) -> tuple[str, list[dict[str, object]]]:
        """List published releases for a public GitHub repository."""

        try:
            results = service.list_releases(owner, repository, max_results)
        except (GitHubServiceError, ValueError) as exc:
            raise ToolException(str(exc)) from exc

        return _format_optional_results(results, "No GitHub releases were found.")

    def search_github_code(
        query: str,
        repository: str | None = None,
        max_results: int = 5,
    ) -> tuple[str, list[dict[str, object]]]:
        """Search public GitHub code using an explicitly configured token."""

        try:
            results = service.search_code(query, repository, max_results)
        except (GitHubServiceError, ValueError) as exc:
            raise ToolException(str(exc)) from exc

        return _format_optional_results(results, "No GitHub code results were found.")

    return [
        StructuredTool.from_function(
            func=search_github_repositories,
            name="search_github_repositories",
            description="Find public GitHub repositories using read-only metadata search.",
            args_schema=_RepositorySearchInput,
            response_format="content_and_artifact",
            handle_tool_error=True,
            handle_validation_error="Use a non-empty query and 1 to 10 results.",
        ),
        StructuredTool.from_function(
            func=read_github_readme,
            name="read_github_readme",
            description="Read the README of a known public GitHub repository.",
            args_schema=_ReadmeInput,
            response_format="content_and_artifact",
            handle_tool_error=True,
            handle_validation_error="Use valid GitHub owner, repository, and ref values.",
        ),
        StructuredTool.from_function(
            func=list_github_releases,
            name="list_github_releases",
            description="List published releases of a known public GitHub repository.",
            args_schema=_ReleaseInput,
            response_format="content_and_artifact",
            handle_tool_error=True,
            handle_validation_error="Use valid repository values and 1 to 10 results.",
        ),
        StructuredTool.from_function(
            func=search_github_code,
            name="search_github_code",
            description=(
                "Search code in public GitHub repositories. Requires GITHUB_TOKEN and is read-only."
            ),
            args_schema=_CodeSearchInput,
            response_format="content_and_artifact",
            handle_tool_error=True,
            handle_validation_error="Use a non-empty query and an optional owner/repo.",
        ),
    ]


def _run_result_tool(
    operation: Callable[[str, int], list[SearchResultItem]],
    query: str,
    max_results: int,
) -> tuple[str, list[dict[str, object]]]:
    try:
        results = operation(query, max_results)
    except (GitHubServiceError, ValueError) as exc:
        raise ToolException(str(exc)) from exc

    return _format_optional_results(results, "No GitHub repositories were found.")


def _format_optional_results(
    results: Sequence[SearchResultItem],
    empty_message: str,
) -> tuple[str, list[dict[str, object]]]:
    if not results:
        return empty_message, []

    return format_source_tool_results(results)


def _repository_to_search_result(
    repository: _GitHubRepository,
    index: int,
) -> SearchResultItem:
    topics = ", ".join(repository.topics) if repository.topics else "not specified"
    content = "\n".join(
        [
            f"Repository: {repository.full_name}",
            f"Description: {repository.description or 'not provided'}",
            f"Language: {repository.language or 'not specified'}",
            f"Stars: {repository.stargazers_count}",
            f"Default branch: {repository.default_branch}",
            f"Topics: {topics}",
            f"Homepage: {repository.homepage or 'not provided'}",
            f"Updated at: {repository.updated_at or 'not provided'}",
        ]
    )
    provider_score = repository.score if repository.score is not None else 0.0
    score = max(min(provider_score, 1.0), 1.0 / (index + 1))

    return SearchResultItem(
        source=SourceItem(
            title=repository.full_name,
            url=repository.html_url,
            source_type="github",
        ),
        content=content[:MAX_GITHUB_CONTENT_LENGTH],
        score=min(score, 1.0),
    )


def _release_to_search_result(release: _GitHubRelease, index: int) -> SearchResultItem:
    release_name = release.name or release.tag_name
    content = "\n".join(
        [
            f"Release: {release_name}",
            f"Tag: {release.tag_name}",
            f"Published at: {release.published_at or 'not provided'}",
            f"Prerelease: {release.prerelease}",
            f"Notes: {release.body or 'No release notes were provided.'}",
        ]
    )

    return SearchResultItem(
        source=SourceItem(
            title=release_name,
            url=release.html_url,
            source_type="release_notes",
        ),
        content=content[:MAX_GITHUB_CONTENT_LENGTH],
        score=1.0 / (index + 1),
    )


def _code_item_to_search_result(item: _GitHubCodeItem, index: int) -> SearchResultItem:
    fragments = "\n\n".join(match.fragment for match in item.text_matches)
    content = "\n".join(
        [
            f"Repository: {item.repository.full_name}",
            f"Path: {item.path}",
            f"Matched content: {fragments or 'No text fragment was returned.'}",
        ]
    )

    return SearchResultItem(
        source=SourceItem(
            title=f"{item.repository.full_name}: {item.path}",
            url=item.html_url,
            source_type="github",
        ),
        content=content[:MAX_GITHUB_CONTENT_LENGTH],
        score=1.0 / (index + 1),
    )


def _validate_json_response(
    response: httpx.Response,
    model_type: type[ModelT],
) -> ModelT:
    try:
        return model_type.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise InvalidGitHubResponseError("GitHub returned invalid response data") from exc


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise GitHubAuthenticationError("GitHub rejected the configured token")

    if response.status_code == 429:
        raise GitHubRateLimitError("GitHub API rate limit was reached")

    if response.status_code == 403:
        if response.headers.get("x-ratelimit-remaining") == "0" or response.headers.get(
            "retry-after"
        ):
            raise GitHubRateLimitError("GitHub API rate limit was reached")

        raise GitHubPermissionError("GitHub denied access to this resource")

    if response.status_code == 404:
        raise GitHubNotFoundError("GitHub resource was not found")

    if response.status_code == 422:
        raise GitHubInvalidRequestError("GitHub rejected the request parameters")

    if response.is_error:
        raise GitHubUnavailableError("GitHub API returned an error")


def _normalize_token(token: SecretStr | None) -> SecretStr | None:
    if token is None or not token.get_secret_value().strip():
        return None

    return token


def _normalize_query(query: str, label: str) -> str:
    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError(f"{label} must not be empty")

    return normalized_query


def _validate_max_results(max_results: int) -> None:
    if not 1 <= max_results <= MAX_GITHUB_RESULTS:
        raise ValueError(f"max_results must be between 1 and {MAX_GITHUB_RESULTS}")


def _validate_repository_part(value: str, label: str) -> str:
    normalized_value = value.strip()

    if not normalized_value or not REPOSITORY_PART_PATTERN.fullmatch(normalized_value):
        raise ValueError(f"GitHub {label} is invalid")

    return normalized_value


def _validate_repository_name(repository: str) -> str:
    parts = repository.strip().split("/")

    if len(parts) != 2:
        raise ValueError("GitHub repository must use owner/repository format")

    owner = _validate_repository_part(parts[0], "owner")
    repository_name = _validate_repository_part(parts[1], "repository")
    return f"{owner}/{repository_name}"


def _copy_results(results: Sequence[SearchResultItem]) -> list[SearchResultItem]:
    return [item.model_copy(deep=True) for item in results]
