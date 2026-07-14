from typing import Any, TypeAlias

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from ai_research_assistant.agents import create_research_agent
from ai_research_assistant.config import get_settings
from ai_research_assistant.graph.nodes import (
    ResearchRunner,
    ResponseGenerator,
    RouteClassifier,
    create_clarification_node,
    create_research_node,
    create_response_node,
    create_router_node,
)
from ai_research_assistant.graph.ollama import (
    create_ollama_response_generator,
    create_ollama_route_classifier,
)
from ai_research_assistant.graph.state import AppState
from ai_research_assistant.llm import create_chat_model
from ai_research_assistant.memory import create_sqlite_checkpointer

CoreGraph: TypeAlias = CompiledStateGraph[
    AppState,
    None,
    AppState,
    AppState,
]


def build_core_graph(
    classify: RouteClassifier,
    generate: ResponseGenerator,
    research: ResearchRunner,
    checkpointer: Checkpointer = None,
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
    research_node: RunnableLambda[AppState, Any] = RunnableLambda(create_research_node(research))
    clarification: RunnableLambda[AppState, Any] = RunnableLambda(create_clarification_node())
    route_unavailable: RunnableLambda[AppState, Any] = RunnableLambda(
        create_response_node("route_unavailable", generate)
    )

    builder.add_node(
        "router",
        router,
        destinations=(
            "direct_answer",
            "unsupported",
            "research",
            "clarification",
            "route_unavailable",
        ),
    )
    builder.add_node("direct_answer", direct_answer)
    builder.add_node("unsupported", unsupported)
    builder.add_node("research", research_node)
    builder.add_node("clarification", clarification, destinations=("router",))
    builder.add_node("route_unavailable", route_unavailable)

    builder.add_edge(START, "router")

    builder.add_edge("direct_answer", END)
    builder.add_edge("unsupported", END)
    builder.add_edge("research", END)
    builder.add_edge("route_unavailable", END)

    return builder.compile(checkpointer=checkpointer)


def build_app_graph() -> CoreGraph:
    """Build the application graph with configured live dependencies."""

    model = create_chat_model()
    research_agent = create_research_agent(model=model)
    checkpointer = create_sqlite_checkpointer(get_settings().checkpoint_db_path)

    return build_core_graph(
        classify=create_ollama_route_classifier(model),
        generate=create_ollama_response_generator(model),
        research=research_agent.run,
        checkpointer=checkpointer,
    )
