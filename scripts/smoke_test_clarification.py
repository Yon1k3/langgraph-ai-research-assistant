from uuid import uuid4

from langgraph.types import Command

from ai_research_assistant.graph import CoreGraph, open_app_graph


def main() -> None:
    """Run one live clarification interrupt and resume through Ollama."""

    print("Building the checkpointed application graph...")

    with open_app_graph() as graph:
        run_clarification_check(graph)


def run_clarification_check(graph: CoreGraph) -> None:
    """Run pause and resume assertions against one managed graph."""

    config = {
        "configurable": {
            "thread_id": f"clarification-smoke-{uuid4()}",
        }
    }
    query = "Допоможи мені з агентом."

    print(f"\nQuery: {query}")
    interrupted = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
        config=config,
    )
    interrupts = interrupted.get("__interrupt__", [])

    if len(interrupts) != 1:
        raise RuntimeError(
            "Expected exactly one clarification interrupt, "
            f"got route={interrupted.get('route')!r}, "
            f"reason={interrupted.get('routing_reason')!r}"
        )

    payload = interrupts[0].value

    if not isinstance(payload, dict) or payload.get("type") != "clarification":
        raise RuntimeError("Clarification interrupt returned an invalid payload")

    print(f"Question: {payload.get('question')}")

    clarification = "Я створюю LangGraph-агента з вебпошуком і хочу зрозуміти архітектуру."
    print(f"Clarification: {clarification}")

    resumed = graph.invoke(
        Command(resume=clarification),
        config=config,
    )

    if resumed.get("__interrupt__"):
        raise RuntimeError("Graph requested another unexpected clarification")

    print(f"Final route: {resumed['route']}")
    print(f"Response: {resumed['messages'][-1].content}")
    print("\nClarification smoke test passed.")


if __name__ == "__main__":
    main()
