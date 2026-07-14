from collections.abc import Callable
from typing import Literal, TypeAlias

from langgraph.types import Command

from ai_research_assistant.graph.state import AppState
from ai_research_assistant.models import (
    ResearchResult,
    RouteDecision,
    RouteName,
)

CoreNodeName: TypeAlias = Literal[
    "direct_answer",
    "unsupported",
    "research",
    "route_unavailable",
]
ResponseKind: TypeAlias = Literal[
    "direct_answer",
    "unsupported",
    "route_unavailable",
]

RouteClassifier: TypeAlias = Callable[[str], RouteDecision]
ResponseGenerator: TypeAlias = Callable[[str, str, ResponseKind], str]
ResearchRunner: TypeAlias = Callable[[str, str], ResearchResult]
NodeUpdate: TypeAlias = dict[str, object]


def get_latest_user_text(state: AppState) -> str:
    """Return the text of the latest user message."""

    for message in reversed(state["messages"]):
        if message.type != "human":
            continue

        if not isinstance(message.content, str):
            raise TypeError("The latest user message must contain text")

        return message.content

    raise ValueError("The graph state does not contain a user message")


def resolve_destination(route: RouteName) -> CoreNodeName:
    """Map a routing decision to an implemented graph node."""

    if route == "direct_answer":
        return "direct_answer"

    if route == "unsupported":
        return "unsupported"

    if route == "research":
        return "research"

    return "route_unavailable"


def create_router_node(
    classify: RouteClassifier,
) -> Callable[[AppState], Command[CoreNodeName]]:
    """Create a router node using the provided classifier."""

    def router_node(state: AppState) -> Command[CoreNodeName]:
        query = get_latest_user_text(state)
        decision = classify(query)

        return Command(
            goto=resolve_destination(decision.route),
            update={
                "route": decision.route,
                "routing_reason": decision.reason,
                "routing_confidence": decision.confidence,
                "response_language": decision.response_language,
                "clarification_question": decision.clarification_question,
            },
        )

    return router_node


def create_response_node(
    kind: ResponseKind,
    generate: ResponseGenerator,
) -> Callable[[AppState], NodeUpdate]:
    """Create a terminal response node."""

    def response_node(state: AppState) -> NodeUpdate:
        query = get_latest_user_text(state)
        response = generate(
            query,
            state["response_language"],
            kind,
        )

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": response,
                }
            ],
            "sources": [],
        }

    return response_node


def create_research_node(
    research: ResearchRunner,
) -> Callable[[AppState], NodeUpdate]:
    """Create a terminal node backed by the Research Agent."""

    def research_node(state: AppState) -> NodeUpdate:
        query = get_latest_user_text(state)
        result = research(
            query,
            state["response_language"],
        )

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": result.answer,
                }
            ],
            "sources": result.sources,
        }

    return research_node
