from ai_research_assistant.tools import create_tavily_search_service


def main() -> None:
    """Run one live search against the configured Tavily API."""

    print("Connecting to Tavily...")
    service = create_tavily_search_service()

    results = service.search(
        "LangGraph official documentation and latest releases",
        max_results=3,
    )

    if not results:
        raise RuntimeError("Tavily returned no search results")

    for index, item in enumerate(results, start=1):
        print(f"\n[{index}] {item.source.title}")
        print(f"Type:  {item.source.source_type}")
        print(f"URL:   {item.source.url}")
        print(f"Score: {item.score:.3f}")

    print("\nTavily search smoke test passed.")


if __name__ == "__main__":
    main()
