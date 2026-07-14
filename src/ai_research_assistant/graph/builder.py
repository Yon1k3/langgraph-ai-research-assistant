from typing import Any, TypeAlias

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_research_assistant.graph.nodes import (
    ResponseGenerator,
    RouteClassifier,
    create_response_node,
    create_router_node,
)
from ai_research_assistant.graph.state import AppState

CoreGraph: TypeAlias = CompiledStateGraph[
    AppState,
    None,
    AppState,
    AppState,
]


def build_core_graph(
    classify: RouteClassifier,
    generate: ResponseGenerator,
) -> CoreGraph:
    """Build and compile the core application graph."""

    builder = StateGraph(AppState)

    router: RunnableLambda[AppState, Any] = RunnableLambda(create_router_node(classify))
    direct_answer: RunnableLambda[AppState, Any] = RunnableLambda(
        create_response_node("direct_answer", generate)
    )
    unsupported: RunnableLambda[AppState, Any] = RunnableLambda(
        create_response_node("unsupported", generate)
    )
    route_unavailable: RunnableLambda[AppState, Any] = RunnableLambda(
        create_response_node("route_unavailable", generate)
    )

    builder.add_node(
        "router",
        router,
        destinations=(
            "direct_answer",
            "unsupported",
            "route_unavailable",
        ),
    )
    builder.add_node("direct_answer", direct_answer)
    builder.add_node("unsupported", unsupported)
    builder.add_node("route_unavailable", route_unavailable)

    builder.add_edge(START, "router")

    builder.add_edge("direct_answer", END)
    builder.add_edge("unsupported", END)
    builder.add_edge("route_unavailable", END)

    return builder.compile()
