from langgraph.graph import MessagesState

from ai_research_assistant.models import AgentResultRecord, ErrorInfoRecord, RouteName


class AppState(MessagesState, total=False):
    """Shared state of the application graph."""

    route: RouteName | None
    routing_reason: str | None
    routing_confidence: float | None
    response_language: str
    clarification_question: str | None
    agent_result: AgentResultRecord | None
    error: ErrorInfoRecord | None
