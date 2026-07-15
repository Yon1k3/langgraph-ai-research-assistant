from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from langgraph.types import Command, interrupt

from ai_research_assistant.conversation import (
    ConversationContext,
    build_conversation_context,
)
from ai_research_assistant.errors import (
    InvalidModelOutputError,
    detect_fallback_language,
    map_runtime_error,
)
from ai_research_assistant.graph.state import AppState
from ai_research_assistant.models import (
    AgentResult,
    CodeResult,
    ErrorInfo,
    ResearchResult,
    RouteDecision,
    RouteName,
)

CoreNodeName: TypeAlias = Literal[
    "direct_answer",
    "unsupported",
    "research",
    "code",
    "clarification",
    "route_unavailable",
    "error",
    "finalize",
]
AgentNodeDestination: TypeAlias = Literal["finalize", "error"]
ResponseKind: TypeAlias = Literal[
    "direct_answer",
    "unsupported",
    "route_unavailable",
]

RouteClassifier: TypeAlias = Callable[[ConversationContext], RouteDecision]
ResponseGenerator: TypeAlias = Callable[[ConversationContext, str, ResponseKind], str]
ResearchRunner: TypeAlias = Callable[[str, str], ResearchResult]
CodeRunner: TypeAlias = Callable[[str, str], CodeResult]
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

    if route == "code":
        return "code"

    if route == "clarification":
        return "clarification"

    return "route_unavailable"


def create_router_node(
    classify: RouteClassifier,
) -> Callable[[AppState], Command[CoreNodeName]]:
    """Create a router node using the provided classifier."""

    def router_node(state: AppState) -> Command[CoreNodeName]:
        context = build_conversation_context(state["messages"])
        query = context.latest_user_query

        try:
            decision = classify(context)
        except Exception as exc:
            return _create_runtime_error_command(
                exc,
                detect_fallback_language(query),
                clear_routing=True,
            )

        return Command(
            goto=resolve_destination(decision.route),
            update={
                "route": decision.route,
                "routing_reason": decision.reason,
                "routing_confidence": decision.confidence,
                "response_language": decision.response_language,
                "resolved_query": decision.resolved_query,
                "clarification_question": decision.clarification_question,
                "agent_result": None,
                "error": None,
            },
        )

    return router_node


def create_response_node(
    kind: ResponseKind,
    generate: ResponseGenerator,
) -> Callable[[AppState], Command[AgentNodeDestination]]:
    """Create a response node that writes a canonical agent result."""

    def response_node(state: AppState) -> Command[AgentNodeDestination]:
        context = build_conversation_context(state["messages"])
        language = state["response_language"]

        try:
            response = generate(context, language, kind)

            if not isinstance(response, str) or not response.strip():
                raise InvalidModelOutputError("Response generator returned no usable text")

            result = AgentResult(answer=response)
        except Exception as exc:
            return _create_runtime_error_command(exc, language)

        return Command(
            goto="finalize",
            update={
                "agent_result": result.to_record(),
                "error": None,
            },
        )

    return response_node


def create_research_node(
    research: ResearchRunner,
) -> Callable[[AppState], Command[AgentNodeDestination]]:
    """Create a Research Agent node that writes a canonical agent result."""

    def research_node(state: AppState) -> Command[AgentNodeDestination]:
        resolved_query = state.get("resolved_query")
        language = state["response_language"]

        if not resolved_query:
            raise ValueError("Research node requires a resolved query")

        try:
            result = research(resolved_query, language)

            if not isinstance(result, ResearchResult):
                raise InvalidModelOutputError("Research runner returned an unexpected result type")

            result_record = result.to_record()
        except Exception as exc:
            return _create_runtime_error_command(exc, language)

        return Command(
            goto="finalize",
            update={
                "agent_result": result_record,
                "error": None,
            },
        )

    return research_node


def create_code_node(
    code: CodeRunner,
) -> Callable[[AppState], Command[AgentNodeDestination]]:
    """Create a Code Agent node that writes a canonical agent result."""

    def code_node(state: AppState) -> Command[AgentNodeDestination]:
        resolved_query = state.get("resolved_query")
        language = state["response_language"]

        if not resolved_query:
            raise ValueError("Code node requires a resolved query")

        try:
            result = code(resolved_query, language)

            if not isinstance(result, CodeResult):
                raise InvalidModelOutputError("Code runner returned an unexpected result type")

            result_record = result.to_record()
        except Exception as exc:
            return _create_runtime_error_command(exc, language)

        return Command(
            goto="finalize",
            update={
                "agent_result": result_record,
                "error": None,
            },
        )

    return code_node


def create_error_node() -> Callable[[AppState], NodeUpdate]:
    """Convert safe runtime error metadata into a canonical agent result."""

    def error_node(state: AppState) -> NodeUpdate:
        error_record = state.get("error")

        if error_record is None:
            raise ValueError("Error node requires safe error metadata")

        error = ErrorInfo.model_validate(error_record)
        result = AgentResult(answer=error.message)

        return {
            "agent_result": result.to_record(),
            "clarification_question": None,
        }

    return error_node


def create_finalize_node() -> Callable[[AppState], NodeUpdate]:
    """Append the already validated result without inventing new content."""

    def finalize_node(state: AppState) -> NodeUpdate:
        result_record = state.get("agent_result")

        if result_record is None:
            raise ValueError("Finalize node requires an agent result")

        result = AgentResult.from_record(result_record)

        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": result.answer,
                }
            ]
        }

    return finalize_node


def create_clarification_node() -> Callable[[AppState], Command[Literal["router"]]]:
    """Create a node that pauses for clarification and returns to the router."""

    def clarification_node(state: AppState) -> Command[Literal["router"]]:
        question = state.get("clarification_question")

        if not question:
            raise ValueError("Clarification node requires a question")

        prompt = {
            "type": "clarification",
            "question": question,
        }

        while True:
            answer = interrupt(prompt)

            if isinstance(answer, str) and answer.strip():
                normalized_answer = answer.strip()
                break

            prompt = {
                "type": "clarification",
                "question": question,
                "error": (
                    "Будь ласка, надай непорожню текстову відповідь."
                    if state.get("response_language") == "uk"
                    else "Please provide a non-empty text answer."
                ),
            }

        return Command(
            goto="router",
            update={
                "messages": [
                    {
                        "role": "user",
                        "content": normalized_answer,
                    }
                ],
                "clarification_question": None,
                "resolved_query": None,
                "agent_result": None,
                "error": None,
            },
        )

    return clarification_node


def _create_runtime_error_command(
    error: Exception,
    language: str,
    *,
    clear_routing: bool = False,
) -> Command[Any]:
    error_info = map_runtime_error(error, language)

    if error_info is None:
        raise error

    update: NodeUpdate = {
        "agent_result": None,
        "error": error_info.to_record(),
    }

    if clear_routing:
        update.update(
            {
                "route": None,
                "routing_reason": None,
                "routing_confidence": None,
                "response_language": language,
                "resolved_query": None,
                "clarification_question": None,
            }
        )

    return Command(goto="error", update=update)
