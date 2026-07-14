from ai_research_assistant.graph import build_app_graph
from ai_research_assistant.models import RouteName

TEST_CASES: tuple[tuple[str, RouteName], ...] = (
    ("Привіт!", "direct_answer"),
    ("Склади план харчування на тиждень.", "unsupported"),
    (
        "Поясни актуальні можливості LangGraph і наведи офіційні джерела.",
        "research",
    ),
)


def main() -> None:
    """Run live application graph checks with Ollama and Tavily."""

    print("Building the live application graph...")
    graph = build_app_graph()

    for query, expected_route in TEST_CASES:
        print(f"\nQuery: {query}")

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            }
        )

        actual_route = result["route"]
        response = result["messages"][-1].content
        sources = result.get("sources", [])

        print(f"Expected route: {expected_route}")
        print(f"Actual route:   {actual_route}")
        print(f"Confidence:     {result['routing_confidence']}")
        print(f"Reason:         {result['routing_reason']}")
        print(f"Response:       {response}")

        if actual_route != expected_route:
            raise RuntimeError(f"Expected route {expected_route}, got {actual_route}")

        if expected_route == "research":
            print("Verified sources:")

            for index, source in enumerate(sources, start=1):
                print(f"[{index}] {source.title}")
                print(f"    Type: {source.source_type}")
                print(f"    URL:  {source.url}")

            if not sources:
                raise RuntimeError("Research route returned no verified sources")

            if not any("langgraph" in f"{source.title} {source.url}".lower() for source in sources):
                raise RuntimeError("Research route returned no LangGraph-related sources")
        elif sources:
            raise RuntimeError(f"Route {expected_route} returned unexpected sources")

    print("\nAll live application graph smoke tests passed.")


if __name__ == "__main__":
    main()
