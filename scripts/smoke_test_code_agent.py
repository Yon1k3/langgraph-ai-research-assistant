from ai_research_assistant.agents import create_code_agent

QUERY = "Покажи мінімальний Python-приклад правильного імпорту StateGraph і START з LangGraph."
OFFICIAL_SOURCE_PREFIXES = (
    "https://docs.langchain.com/",
    "https://reference.langchain.com/",
    "https://github.com/langchain-ai/langgraph",
)


def main() -> None:
    """Run one live Code Agent check without executing generated code."""

    print("Building the Code Agent...")
    print(f"\nQuery: {QUERY}")

    result = create_code_agent().run(QUERY, response_language="uk")

    print(f"\nAnswer:\n{result.answer}")
    print("\nVerified sources:")

    for index, source in enumerate(result.sources, start=1):
        print(f"[{index}] {source.title}")
        print(f"    Type: {source.source_type}")
        print(f"    URL:  {source.url}")

    if not result.sources:
        raise RuntimeError("Code Agent returned no verified sources")

    if "```python" not in result.answer:
        raise RuntimeError("Code Agent returned no Python code block")

    if "from langgraph.graph import" not in result.answer:
        raise RuntimeError("Code Agent returned an unsupported import path")

    if "StateGraph" not in result.answer or "START" not in result.answer:
        raise RuntimeError("Code Agent omitted a requested import")

    if any(not str(source.url).startswith(OFFICIAL_SOURCE_PREFIXES) for source in result.sources):
        raise RuntimeError("Code Agent returned a non-official source")

    answer_lines = {line.strip() for line in result.answer.splitlines()}

    if "StateGraphTool" in result.answer or "import StateGraph" in answer_lines:
        raise RuntimeError("Code Agent returned a known hallucinated API")

    print("\nCode Agent smoke test passed. Generated code was not executed.")


if __name__ == "__main__":
    main()
