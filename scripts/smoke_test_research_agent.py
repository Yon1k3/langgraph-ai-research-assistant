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

    if not any("langgraph" in f"{source.title} {source.url}".lower() for source in result.sources):
        raise RuntimeError("Research Agent returned no LangGraph-related sources")

    official_source_prefixes = (
        "https://docs.langchain.com/",
        "https://github.com/langchain-ai/langgraph",
        "https://langchain-ai.github.io/langgraph",
    )
    unexpected_sources = [
        str(source.url)
        for source in result.sources
        if not str(source.url).startswith(official_source_prefixes)
    ]

    if unexpected_sources:
        raise RuntimeError(
            "Research Agent returned non-official sources for an official-source request: "
            + ", ".join(unexpected_sources)
        )

    normalized_answer = result.answer.casefold()
    known_bad_claims = (
        "версія langchain",
        "версією langchain",
        "бібліотека для обробки природної мови",
        "файлова система",
        "langsmith hub",
        "langchain має",
        "оркестровка",
        "в лупі",
    )
    detected_bad_claims = [claim for claim in known_bad_claims if claim in normalized_answer]

    if detected_bad_claims:
        raise RuntimeError(
            "Research Agent returned a known grounding regression: "
            + ", ".join(detected_bad_claims)
        )

    print("\nResearch Agent smoke test passed.")


if __name__ == "__main__":
    main()
