import pytest
from pydantic import ValidationError

from ai_research_assistant.models import RouteDecision, RouteName

ALL_ROUTES: tuple[RouteName, ...] = (
    "research",
    "code",
    "comparison",
    "direct_answer",
    "clarification",
    "unsupported",
)


@pytest.mark.parametrize("route", ALL_ROUTES)
def test_route_decision_accepts_supported_routes(route: RouteName) -> None:
    clarification_question = (
        "Which technology do you want to research?" if route == "clarification" else None
    )

    decision = RouteDecision(
        route=route,
        confidence=0.8,
        reason="The request matches this route.",
        response_language="en",
        clarification_question=clarification_question,
    )

    assert decision.route == route


@pytest.mark.parametrize("confidence", (-0.1, 1.1))
def test_route_decision_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            route="direct_answer",
            confidence=confidence,
            reason="Greeting detected.",
            response_language="en",
        )


def test_clarification_route_requires_question() -> None:
    with pytest.raises(
        ValidationError,
        match="clarification_question is required",
    ):
        RouteDecision(
            route="clarification",
            confidence=0.6,
            reason="The request is ambiguous.",
            response_language="en",
        )


def test_other_routes_reject_clarification_question() -> None:
    with pytest.raises(
        ValidationError,
        match="clarification_question is only allowed",
    ):
        RouteDecision(
            route="research",
            confidence=0.9,
            reason="Research is required.",
            response_language="en",
            clarification_question="What exactly do you mean?",
        )
