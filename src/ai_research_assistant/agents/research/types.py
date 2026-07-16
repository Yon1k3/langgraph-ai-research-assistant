from collections.abc import Callable
from typing import Protocol

from ai_research_assistant.models import EvidenceItem, ResearchSynthesis, SearchResultItem


class SearchService(Protocol):
    """Interface required by the research search tool."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]:
        """Return normalized search results."""


class AgentRunner(Protocol):
    """Minimal interface required from a compiled LangChain agent."""

    def invoke(self, input: dict[str, object]) -> dict[str, object]:
        """Run the agent with a new message state."""


EvidenceSynthesizer = Callable[[str, str, list[EvidenceItem]], ResearchSynthesis]
