from ai_research_assistant.agents import create_research_agent


def main() -> None:
    """Run one live research request through Ollama and Tavily."""

    print("Building the Research Agent...")
    agent = create_research_agent()

    query = (
        "Поясни актуальне призначення LangGraph, його основні можливості "
        "та наведи інформацію з офіційних джерел."
    )

    print(f"\nQuery: {query}")
    result = agent.run(query, response_language="uk")

    print("\nAnswer:")
    print(result.answer)

    print("\nVerified sources:")

    for index, source in enumerate(result.sources, start=1):
        print(f"[{index}] {source.title}")
        print(f"    Type: {source.source_type}")
        print(f"    URL:  {source.url}")

    if not result.sources:
        raise RuntimeError("Research Agent returned no verified sources")

    print("\nResearch Agent smoke test passed.")


if __name__ == "__main__":
    main()
