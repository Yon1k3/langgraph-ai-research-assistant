from collections.abc import Callable
from typing import Protocol, TypeAlias

from langchain_core.messages import ToolCall, ToolMessage

from ai_research_assistant.models import CodeSynthesis, EvidenceItem


class AgentRunner(Protocol):
    """Minimal interface required from a compiled LangChain agent."""

    def invoke(
        self,
        input: dict[str, object],
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Run the agent with a new message state."""


CodeSynthesizer = Callable[[str, str, list[EvidenceItem]], CodeSynthesis]
RepositoryCandidate: TypeAlias = tuple[str, ToolCall, ToolMessage, EvidenceItem]
