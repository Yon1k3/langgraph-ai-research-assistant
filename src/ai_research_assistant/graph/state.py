from langgraph.graph import MessagesState

from ai_research_assistant.models import RouteName, SourceItem


class AppState(MessagesState, total=False):
    """Shared state of the application graph."""

    route: RouteName
    routing_reason: str
    routing_confidence: float
    response_language: str
    clarification_question: str | None
    sources: list[SourceItem]
