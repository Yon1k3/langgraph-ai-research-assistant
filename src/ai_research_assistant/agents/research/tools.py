from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from ai_research_assistant.agents.research.constants import MAX_AGENT_EVIDENCE_CONTENT_LENGTH
from ai_research_assistant.agents.research.evidence import (
    _create_evidence_item,
    _select_relevant_excerpt,
)
from ai_research_assistant.agents.research.types import SearchService
from ai_research_assistant.tools import format_search_service_error
from ai_research_assistant.tools.web_search import SearchServiceError


class _WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _format_search_tool_error(error: ToolException) -> str:
    cause = error.__cause__

    if isinstance(cause, SearchServiceError):
        return format_search_service_error(cause)

    return f"Web search failed: {error}"


def create_web_search_tool(search_service: SearchService) -> BaseTool:
    """Create a read-only web search tool with separate source artifacts."""

    def search_web(
        query: str,
        max_results: int = 5,
    ) -> tuple[str, list[dict[str, object]]]:
        """Search the web for current technical information and official sources."""

        try:
            results = search_service.search(query=query, max_results=max_results)
        except (SearchServiceError, ValueError) as exc:
            raise ToolException(str(exc)) from exc

        if not results:
            return "No search results were found for this query.", []

        content_parts: list[str] = []
        evidence_artifacts: list[dict[str, object]] = []

        for item in results:
            evidence = _create_evidence_item(item)
            content_parts.append(
                "\n".join(
                    [
                        f"[{evidence.source_id}] {item.source.title}",
                        f"URL: {item.source.url}",
                        f"Source type: {item.source.source_type}",
                        f"Relevance score: {item.score:.3f}",
                        "Content: "
                        + _select_relevant_excerpt(
                            item.content,
                            query,
                            MAX_AGENT_EVIDENCE_CONTENT_LENGTH,
                        ),
                    ]
                )
            )
            evidence_artifacts.append(evidence.model_dump(mode="json"))

        return "\n\n".join(content_parts), evidence_artifacts

    return StructuredTool.from_function(
        func=search_web,
        name="search_web",
        description=(
            "Search for current technical information. Prefer queries targeting "
            "official documentation, official GitHub repositories, changelogs, "
            "and release notes."
        ),
        args_schema=_WebSearchInput,
        response_format="content_and_artifact",
        handle_tool_error=_format_search_tool_error,
        handle_validation_error=("Use a non-empty search query and max_results between 1 and 10."),
    )
