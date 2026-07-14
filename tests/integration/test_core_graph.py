import pytest

from ai_research_assistant.graph.builder import build_core_graph
from ai_research_assistant.graph.nodes import (
    ResponseKind,
    RouteClassifier,
)
from ai_research_assistant.models import RouteDecision, RouteName


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


@pytest.mark.parametrize(
    ("route", "expected_response_kind"),
    (
        ("direct_answer", "direct_answer"),
        ("unsupported", "unsupported"),
        ("research", "route_unavailable"),
    ),
)
def test_core_graph_routes_to_expected_node(
    route: RouteName,
    expected_response_kind: ResponseKind,
) -> None:
    graph = build_core_graph(
        classify=create_fake_classifier(route),
        generate=fake_response_generator,
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

    assert result["route"] == route
    assert result["routing_confidence"] == 0.9
    assert result["response_language"] == "en"
    assert result["messages"][-1].content == (f"{expected_response_kind}:en:Test request")
