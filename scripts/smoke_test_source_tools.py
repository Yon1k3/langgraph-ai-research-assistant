from ai_research_assistant.config import get_settings
from ai_research_assistant.tools import (
    create_github_search_service,
    create_official_documentation_search_service,
)

OFFICIAL_REPOSITORY = "langchain-ai/langgraph"


def main() -> None:
    """Verify live official documentation and read-only GitHub adapters."""

    print("Searching trusted documentation domain...")
    documentation_service = create_official_documentation_search_service()
    documentation_results = documentation_service.search(
        "LangGraph overview StateGraph",
        ["docs.langchain.com"],
        max_results=3,
    )

    if not documentation_results:
        raise RuntimeError("Official documentation search returned no results")

    for result in documentation_results:
        host = result.source.url.host or ""

        if host != "docs.langchain.com" and not host.endswith(".docs.langchain.com"):
            raise RuntimeError(f"Documentation allowlist leaked domain: {host}")

        print(f"[DOC] {result.source.title}")
        print(f"      {result.source.url}")

    print("\nSearching public GitHub repositories...")
    github_service = create_github_search_service()
    repository_results = github_service.search_repositories(
        "langgraph in:name org:langchain-ai",
        max_results=3,
    )
    official_repository = next(
        (
            result
            for result in repository_results
            if result.source.title.casefold() == OFFICIAL_REPOSITORY
        ),
        None,
    )

    if official_repository is None:
        raise RuntimeError("Official LangGraph repository was not found")

    print(f"[REPO] {official_repository.source.title}")
    print(f"       {official_repository.source.url}")

    readme = github_service.get_readme("langchain-ai", "langgraph")

    if "langgraph" not in readme.content.casefold():
        raise RuntimeError("GitHub README did not contain the repository subject")

    print(f"[README] {readme.source.url}")

    releases = github_service.list_releases(
        "langchain-ai",
        "langgraph",
        max_results=2,
    )

    if not releases:
        raise RuntimeError("GitHub release search returned no published releases")

    print(f"[RELEASE] {releases[0].source.title}")
    print(f"          {releases[0].source.url}")

    github_token = get_settings().github_token

    if github_token is None or not github_token.get_secret_value().strip():
        print("\n[SKIP] GitHub code search requires optional GITHUB_TOKEN.")
    else:
        code_results = github_service.search_code(
            "StateGraph",
            repository=OFFICIAL_REPOSITORY,
            max_results=3,
        )

        if not code_results:
            raise RuntimeError("Authenticated GitHub code search returned no results")

        print(f"\n[CODE] {code_results[0].source.title}")
        print(f"       {code_results[0].source.url}")

    print("\nSource tools smoke test passed.")


if __name__ == "__main__":
    main()
