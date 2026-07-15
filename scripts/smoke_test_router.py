from uuid import uuid4

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
OFFICIAL_LANGGRAPH_SOURCE_PREFIXES = (
    "https://docs.langchain.com/",
    "https://github.com/langchain-ai/langgraph",
    "https://langchain-ai.github.io/langgraph",
)
KNOWN_BAD_RESEARCH_CLAIMS = (
    "версія langchain",
    "версією langchain",
    "бібліотека для обробки природної мови",
    "файлова система",
    "langsmith hub",
    "langchain має",
    "оркестровка",
    "в лупі",
)


def main() -> None:
    """Run live application graph checks with Ollama and Tavily."""

    print("Building the live application graph...")
    graph = build_app_graph()

    for index, (query, expected_route) in enumerate(TEST_CASES, start=1):
        print(f"\nQuery: {query}")

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ]
            },
            config={
                "configurable": {
                    "thread_id": f"router-smoke-{index}-{uuid4()}",
                }
            },
        )

        actual_route = result["route"]
        response = result["messages"][-1].content
        agent_result = result.get("agent_result") or {}
        sources = agent_result.get("sources", [])
        claims = agent_result.get("claims", [])

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
                print(f"[{index}] {source['title']}")
                print(f"    Type: {source['source_type']}")
                print(f"    URL:  {source['url']}")

            if not sources:
                raise RuntimeError("Research route returned no verified sources")

            if not claims:
                raise RuntimeError("Research route returned no grounded claims")

            source_ids = {source["source_id"] for source in sources}

            if any(claim["source_id"] not in source_ids for claim in claims):
                raise RuntimeError("Research route returned an unresolved claim source")

            if not any(
                "langgraph" in f"{source['title']} {source['url']}".lower() for source in sources
            ):
                raise RuntimeError("Research route returned no LangGraph-related sources")

            unexpected_sources = [
                source["url"]
                for source in sources
                if not source["url"].startswith(OFFICIAL_LANGGRAPH_SOURCE_PREFIXES)
            ]

            if unexpected_sources:
                raise RuntimeError(
                    "Research route returned non-official sources: " + ", ".join(unexpected_sources)
                )

            normalized_response = str(response).casefold()
            detected_bad_claims = [
                claim for claim in KNOWN_BAD_RESEARCH_CLAIMS if claim in normalized_response
            ]

            if detected_bad_claims:
                raise RuntimeError(
                    "Research route returned a known grounding regression: "
                    + ", ".join(detected_bad_claims)
                )
        elif sources or claims:
            raise RuntimeError(f"Route {expected_route} returned unexpected evidence")

    print("\nAll live application graph smoke tests passed.")


if __name__ == "__main__":
    main()
