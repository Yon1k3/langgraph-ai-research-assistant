import pytest

from ai_research_assistant.graph.builder import build_core_graph
from ai_research_assistant.graph.nodes import (
    ResponseKind,
    RouteClassifier,
)
from ai_research_assistant.models import (
    ResearchResult,
    RouteDecision,
    RouteName,
    SourceItem,
)


def create_fake_classifier(route: RouteName) -> RouteClassifier:
    """Create a deterministic classifier for an offline graph test."""

    def classify(query: str) -> RouteDecision:
        assert query == "Test request"

        return RouteDecision(
            route=route,
            confidence=0.9,
            reason=f"Selected {route} for the test.",
            response_language="en",
            clarification_question=(
                "What exactly do you want to know?" if route == "clarification" else None
            ),
        )

    return classify


def fake_response_generator(
    query: str,
    language: str,
    kind: ResponseKind,
) -> str:
    """Return a deterministic response without using an LLM."""

    return f"{kind}:{language}:{query}"


def unexpected_research_runner(
    query: str,
    language: str,
) -> ResearchResult:
    """Fail if a non-research route invokes the Research Agent."""

    raise AssertionError(f"Research Agent was called unexpectedly: {language}:{query}")


@pytest.mark.parametrize(
    ("route", "expected_response_kind"),
    (
        ("direct_answer", "direct_answer"),
        ("unsupported", "unsupported"),
        ("code", "route_unavailable"),
    ),
)
def test_core_graph_routes_to_expected_non_research_node(
    route: RouteName,
    expected_response_kind: ResponseKind,
) -> None:
    previous_source = SourceItem(
        title="Previous source",
        url="https://example.com/previous",
        source_type="web",
    )
    graph = build_core_graph(
        classify=create_fake_classifier(route),
        generate=fake_response_generator,
        research=unexpected_research_runner,
    )

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Test request",
                }
            ],
            "sources": [previous_source],
        }
    )

    assert result["route"] == route
    assert result["routing_confidence"] == 0.9
    assert result["response_language"] == "en"
    assert result["messages"][-1].content == (f"{expected_response_kind}:en:Test request")
    assert result["sources"] == []


def test_core_graph_runs_research_agent_and_returns_sources() -> None:
    source = SourceItem(
        title="LangGraph overview",
        url="https://docs.langchain.com/oss/python/langgraph/overview",
        source_type="documentation",
    )

    def fake_research_runner(
        query: str,
        language: str,
    ) -> ResearchResult:
        assert query == "Test request"
        assert language == "en"

        return ResearchResult(
            answer="Research answer",
            sources=[source],
        )

    graph = build_core_graph(
        classify=create_fake_classifier("research"),
        generate=fake_response_generator,
        research=fake_research_runner,
    )

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Test request",
                }
            ]
        }
    )

    assert result["route"] == "research"
    assert result["messages"][-1].content == "Research answer"
    assert result["sources"] == [source]
