import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from ai_research_assistant.agents import (
    InvalidResearchResultError,
    ResearchAgent,
    ResearchSearchError,
    create_web_search_tool,
)
from ai_research_assistant.models import SearchResultItem, SourceItem
from ai_research_assistant.tools.web_search import SearchUnavailableError


class FakeSearchService:
    def __init__(self, results: list[SearchResultItem]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]:
        self.calls.append((query, max_results))
        return self.results


class FailingSearchService:
    def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]:
        raise SearchUnavailableError("Search is unavailable")


class FakeRunner:
    def __init__(self, response_messages: list[BaseMessage]) -> None:
        self.response_messages = response_messages
        self.last_input: dict[str, object] | None = None

    def invoke(self, input: dict[str, object]) -> dict[str, object]:
        self.last_input = input
        input_messages = input.get("messages")

        if not isinstance(input_messages, list):
            raise AssertionError("Runner received no message list")

        return {
            "messages": [
                *input_messages,
                *self.response_messages,
            ]
        }


def make_source() -> SourceItem:
    return SourceItem(
        title="LangGraph overview",
        url="https://docs.langchain.com/oss/python/langgraph/overview",
        source_type="documentation",
    )


def make_search_result(source: SourceItem) -> SearchResultItem:
    return SearchResultItem(
        source=source,
        content="LangGraph is a low-level orchestration framework.",
        score=0.95,
    )


def make_research_agent(
    runner: FakeRunner,
    service: FakeSearchService | None = None,
) -> tuple[ResearchAgent, FakeSearchService]:
    resolved_service = service or FakeSearchService([make_search_result(make_source())])
    search_tool = create_web_search_tool(resolved_service)

    return (
        ResearchAgent(
            runner=runner,
            search_tool=search_tool,
        ),
        resolved_service,
    )


def test_search_tool_returns_content_and_source_artifact() -> None:
    source = make_source()
    service = FakeSearchService([make_search_result(source)])
    search_tool = create_web_search_tool(service)

    message = search_tool.invoke(
        {
            "name": "search_web",
            "args": {
                "query": "LangGraph official documentation",
                "max_results": 3,
            },
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert isinstance(message, ToolMessage)
    assert "LangGraph is a low-level orchestration framework." in message.content
    assert message.artifact == [source.model_dump(mode="json")]
    assert service.calls == [("LangGraph official documentation", 3)]


def test_research_agent_stops_when_initial_search_fails() -> None:
    runner = FakeRunner([AIMessage(content="This answer must not be generated.")])
    search_tool = create_web_search_tool(FailingSearchService())
    agent = ResearchAgent(
        runner=runner,
        search_tool=search_tool,
    )

    with pytest.raises(
        ResearchSearchError,
        match="Search is unavailable",
    ):
        agent.run("Explain LangGraph")

    assert runner.last_input is None


def test_research_agent_stops_when_initial_search_has_no_sources() -> None:
    runner = FakeRunner([AIMessage(content="This answer must not be generated.")])
    agent, _ = make_research_agent(
        runner,
        service=FakeSearchService([]),
    )

    with pytest.raises(
        ResearchSearchError,
        match="no verified sources",
    ):
        agent.run("Explain LangGraph")

    assert runner.last_input is None


def test_search_tool_converts_provider_error_to_tool_error() -> None:
    search_tool = create_web_search_tool(FailingSearchService())

    message = search_tool.invoke(
        {
            "name": "search_web",
            "args": {"query": "LangGraph"},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "Search is unavailable" in message.content


def test_research_agent_runs_initial_search_and_extracts_sources() -> None:
    source = make_source()
    runner = FakeRunner([AIMessage(content="LangGraph is designed for stateful agent workflows.")])
    agent, service = make_research_agent(runner)

    result = agent.run("Explain LangGraph", response_language="en")

    assert result.answer == "LangGraph is designed for stateful agent workflows."
    assert result.sources == [source]
    assert len(service.calls) == 1
    assert service.calls[0][1] == 5
    assert "official documentation" in service.calls[0][0]
    assert runner.last_input is not None


def test_research_agent_focuses_non_english_search_on_technical_terms() -> None:
    runner = FakeRunner([AIMessage(content="LangGraph supports stateful workflows.")])
    agent, service = make_research_agent(runner)

    agent.run(
        "Поясни актуальне призначення LangGraph та наведи офіційні джерела.",
        response_language="uk",
    )

    search_query, _ = service.calls[0]

    assert search_query.startswith("LangGraph official documentation")
    assert "офіційні джерела" not in search_query


def test_research_agent_deduplicates_sources_from_multiple_searches() -> None:
    source = make_source()
    runner = FakeRunner(
        [
            ToolMessage(
                content="Additional search result",
                tool_call_id="call-2",
                name="search_web",
                artifact=[source.model_dump(mode="json")],
            ),
            AIMessage(content="Final answer"),
        ]
    )
    agent, _ = make_research_agent(runner)

    result = agent.run("Explain LangGraph")

    assert result.answer == "Final answer"
    assert result.sources == [source]


def test_research_agent_rejects_missing_final_answer() -> None:
    runner = FakeRunner([])
    agent, _ = make_research_agent(runner)

    with pytest.raises(InvalidResearchResultError, match="no final text answer"):
        agent.run("Explain LangGraph")


def test_research_agent_rejects_invalid_source_artifact() -> None:
    runner = FakeRunner(
        [
            ToolMessage(
                content="Search result",
                tool_call_id="call-2",
                name="search_web",
                artifact=[{"invalid": "source"}],
            ),
            AIMessage(content="Answer"),
        ]
    )
    agent, _ = make_research_agent(runner)

    with pytest.raises(InvalidResearchResultError, match="invalid source metadata"):
        agent.run("Explain LangGraph")


def test_research_agent_rejects_empty_query() -> None:
    runner = FakeRunner([])
    agent, _ = make_research_agent(runner)

    with pytest.raises(ValueError, match="must not be empty"):
        agent.run("   ")


def test_research_agent_removes_verified_url_from_final_answer() -> None:
    source = make_source()
    runner = FakeRunner(
        [
            AIMessage(
                content=(
                    "LangGraph supports stateful workflows.\n\n"
                    "Sources:\n"
                    f"- LangGraph documentation: {source.url}"
                )
            )
        ]
    )
    agent, _ = make_research_agent(runner)

    result = agent.run("Explain LangGraph")

    assert result.answer == "LangGraph supports stateful workflows."
    assert result.sources == [source]
    assert "http" not in result.answer


def test_research_agent_removes_unverified_url_from_final_answer() -> None:
    runner = FakeRunner(
        [
            AIMessage(
                content=(
                    "LangGraph supports stateful workflows.\n\n"
                    "Official sources:\n"
                    "- Invented documentation: https://example.com/invented-documentation\n\n"
                    "Additional details remain available."
                )
            )
        ]
    )
    agent, _ = make_research_agent(runner)

    result = agent.run("Explain LangGraph")

    assert result.answer == (
        "LangGraph supports stateful workflows.\n\nAdditional details remain available."
    )
    assert "http" not in result.answer


def test_research_agent_rejects_answer_containing_only_urls() -> None:
    runner = FakeRunner(
        [
            AIMessage(
                content=(
                    "Sources:\n- Invented documentation: https://example.com/invented-documentation"
                )
            )
        ]
    )
    agent, _ = make_research_agent(runner)

    with pytest.raises(
        InvalidResearchResultError,
        match="no usable text after URL sanitization",
    ):
        agent.run("Explain LangGraph")
