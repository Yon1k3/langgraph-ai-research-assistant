from uuid import uuid4

from ai_research_assistant.graph import open_app_graph

INITIAL_QUERY = "Привіт! Я зараз вивчаю LangGraph."
FOLLOW_UP_QUERY = "А які його мінуси?"


def main() -> None:
    """Verify one live follow-up through Router, Research Agent, and SQLite."""

    config = {
        "configurable": {
            "thread_id": f"conversation-context-smoke-{uuid4()}",
        }
    }

    print("Building the checkpointed application graph...")

    with open_app_graph() as graph:
        first_result = graph.invoke(
            {"messages": [{"role": "user", "content": INITIAL_QUERY}]},
            config=config,
        )
        follow_up_result = graph.invoke(
            {"messages": [{"role": "user", "content": FOLLOW_UP_QUERY}]},
            config=config,
        )

    resolved_query = follow_up_result.get("resolved_query")
    agent_result = follow_up_result.get("agent_result") or {}
    sources = agent_result.get("sources", [])
    claims = agent_result.get("claims", [])

    print(f"Initial query:  {INITIAL_QUERY}")
    print(f"Initial route:  {first_result['route']}")
    print(f"Follow-up:      {FOLLOW_UP_QUERY}")
    print(f"Resolved query: {resolved_query}")
    print(f"Follow-up route:{follow_up_result['route']}")
    print(f"Response:       {follow_up_result['messages'][-1].content}")

    for claim in claims:
        print(f"Supporting quote: {claim['supporting_quote']}")

    for source in sources:
        print(f"Source: {source['title']}")
        print(f"URL:    {source['url']}")

    if follow_up_result["route"] != "research":
        raise RuntimeError("Context-dependent technical follow-up was not routed to research")

    if first_result["route"] != "direct_answer":
        raise RuntimeError("Casual initial message was not handled directly")

    if not isinstance(resolved_query, str) or "langgraph" not in resolved_query.casefold():
        raise RuntimeError("Router did not resolve the LangGraph subject in the follow-up")

    if not sources or not claims:
        raise RuntimeError("Context-dependent research returned no grounded evidence")

    print("\nConversation context smoke test passed.")


if __name__ == "__main__":
    main()
