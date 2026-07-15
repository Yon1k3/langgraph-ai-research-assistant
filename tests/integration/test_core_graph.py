import pytest

from ai_research_assistant.errors import ModelUnavailableError
from ai_research_assistant.graph.builder import build_core_graph
from ai_research_assistant.graph.nodes import (
    ResponseKind,
    RouteClassifier,
)
from ai_research_assistant.models import (
    AgentResult,
    GroundedClaim,
    ResearchResult,
    RouteDecision,
    RouteName,
    SourceItem,
    SourceReference,
)
from ai_research_assistant.tools.web_search import SearchUnavailableError


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
        ("comparison", "route_unavailable"),
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
    previous_result = AgentResult(
        answer="Previous answer",
        sources=[SourceReference.from_source(previous_source)],
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
            "agent_result": previous_result.to_record(),
        }
    )

    assert result["route"] == route
    assert result["routing_confidence"] == 0.9
    assert result["response_language"] == "en"
    assert result["messages"][-1].content == (f"{expected_response_kind}:en:Test request")
    assert result["agent_result"]["sources"] == []
    assert result["agent_result"]["claims"] == []
    assert result["error"] is None


def test_core_graph_runs_research_agent_and_returns_sources() -> None:
    source = SourceItem(
        title="LangGraph overview",
        url="https://docs.langchain.com/oss/python/langgraph/overview",
        source_type="documentation",
    )
    source_reference = SourceReference.from_source(source)
    claim = GroundedClaim(
        statement="LangGraph supports stateful workflows.",
        source_id=source_reference.source_id,
        supporting_quote="LangGraph supports durable stateful agent workflows.",
    )

    def fake_research_runner(
        query: str,
        language: str,
    ) -> ResearchResult:
        assert query == "Test request"
        assert language == "en"

        return ResearchResult(
            answer="Research answer",
            sources=[source_reference],
            claims=[claim],
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
    assert (
        result["agent_result"]
        == ResearchResult(
            answer="Research answer",
            sources=[source_reference],
            claims=[claim],
        ).to_record()
    )
    assert result["error"] is None


def test_core_graph_returns_safe_error_when_model_is_unavailable() -> None:
    def unavailable_classifier(query: str) -> RouteDecision:
        raise ModelUnavailableError(f"Secret provider detail for {query}")

    graph = build_core_graph(
        classify=unavailable_classifier,
        generate=fake_response_generator,
        research=unexpected_research_runner,
    )

    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "Test request"}],
            "route": "research",
            "routing_reason": "Stale routing decision",
            "routing_confidence": 1.0,
        }
    )

    assert result["error"]["category"] == "model_unavailable"
    assert result["route"] is None
    assert result["routing_reason"] is None
    assert result["routing_confidence"] is None
    assert "Secret provider detail" not in result["messages"][-1].content
    assert result["agent_result"]["answer"] == result["messages"][-1].content
    assert result["agent_result"]["sources"] == []


def test_core_graph_returns_safe_error_when_research_search_is_unavailable() -> None:
    def unavailable_research(query: str, language: str) -> ResearchResult:
        raise SearchUnavailableError(f"Private service detail for {language}:{query}")

    graph = build_core_graph(
        classify=create_fake_classifier("research"),
        generate=fake_response_generator,
        research=unavailable_research,
    )

    result = graph.invoke({"messages": [{"role": "user", "content": "Test request"}]})

    assert result["error"]["category"] == "search_unavailable"
    assert "Private service detail" not in result["messages"][-1].content
    assert result["agent_result"]["sources"] == []


def test_core_graph_returns_safe_error_for_invalid_generated_response() -> None:
    def empty_response_generator(
        query: str,
        language: str,
        kind: ResponseKind,
    ) -> str:
        return "   "

    graph = build_core_graph(
        classify=create_fake_classifier("direct_answer"),
        generate=empty_response_generator,
        research=unexpected_research_runner,
    )

    result = graph.invoke({"messages": [{"role": "user", "content": "Test request"}]})

    assert result["error"]["category"] == "invalid_model_output"
    assert result["agent_result"]["answer"] == result["messages"][-1].content


def test_core_graph_does_not_hide_programming_errors() -> None:
    def broken_classifier(query: str) -> RouteDecision:
        raise AssertionError(f"Programming bug for {query}")

    graph = build_core_graph(
        classify=broken_classifier,
        generate=fake_response_generator,
        research=unexpected_research_runner,
    )

    with pytest.raises(AssertionError, match="Programming bug"):
        graph.invoke({"messages": [{"role": "user", "content": "Test request"}]})
