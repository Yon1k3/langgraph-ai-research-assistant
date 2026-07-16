from typing import Protocol, TypeVar

from ai_research_assistant.models import AgentResult

AgentResultT_co = TypeVar(
    "AgentResultT_co",
    bound=AgentResult,
    covariant=True,
)


class SpecialistAgent(Protocol[AgentResultT_co]):
    """Shared contract implemented structurally by every specialist agent."""

    def run(
        self,
        query: str,
        response_language: str = "uk",
    ) -> AgentResultT_co:
        """Process one routed request and return a canonical agent result."""
