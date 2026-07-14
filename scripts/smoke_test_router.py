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
    """Run live routing checks against the configured Ollama model."""

    print("Building the Ollama-backed graph...")
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

        print(f"Expected route: {expected_route}")
        print(f"Actual route:   {actual_route}")
        print(f"Confidence:     {result['routing_confidence']}")
        print(f"Reason:         {result['routing_reason']}")
        print(f"Response:       {response}")

        if actual_route != expected_route:
            raise RuntimeError(f"Expected route {expected_route}, got {actual_route}")

    print("\nAll live router smoke tests passed.")


if __name__ == "__main__":
    main()
