import re
from typing import Protocol, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_research_assistant.llm import create_chat_model
from ai_research_assistant.models import (
    ResearchResult,
    SearchResultItem,
    SourceItem,
)
from ai_research_assistant.tools import create_tavily_search_service
from ai_research_assistant.tools.web_search import SearchServiceError

RESEARCH_SYSTEM_PROMPT = """
You are the Research Agent of a technical AI research assistant.

Your domain is software engineering, artificial intelligence, machine learning,
developer tools, frameworks, libraries, APIs, and technical architecture.

Rules:

- A mandatory initial search_web result is provided in the message history.
- Use that result before writing the final answer.
- Call search_web again only when the initial results are insufficient.
- Base factual and current claims only on successful search_web results.
- Prefer official documentation, official repositories, changelogs, and release notes.
- Clearly distinguish confirmed facts from conclusions or limitations.
- Do not invent URLs, versions, features, quotations, or source titles.
- Respond entirely in the language represented by the ISO code in the user message.
- If no successful search result is available, explain that fresh information could
  not be retrieved. Do not answer from uncertain model memory.
- The final answer must contain only the answer body.
- Never include URLs, Markdown links, or a sources section in the final answer.
  The application renders verified sources separately.
""".strip()

OFFICIAL_SOURCE_HINT = "official documentation official GitHub repository changelog release notes"
TECHNICAL_TERM_PATTERN = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9_.+#-]{1,}(?!\w)")
URL_PATTERN = re.compile(r"https?://[^\s<>]+")
URL_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
SOURCE_HEADING_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(?:(?:official|verified)\s+sources|sources|"
    r"(?:офіційні|перевірені)?\s*джерела)\s*:?\s*$",
    re.IGNORECASE,
)


class SearchService(Protocol):
    """Interface required by the research search tool."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]:
        """Return normalized search results."""


class AgentRunner(Protocol):
    """Minimal interface required from a compiled LangChain agent."""

    def invoke(self, input: dict[str, object]) -> dict[str, object]:
        """Run the agent with a new message state."""


class InvalidResearchResultError(RuntimeError):
    """Raised when the research agent returns an invalid state."""


class ResearchSearchError(RuntimeError):
    """Raised when mandatory research search produces no usable sources."""


class _WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _format_search_tool_error(error: ToolException) -> str:
    return f"Web search failed: {error}"


class ResearchAgent:
    """Run technical research and return an answer with verified sources."""

    def __init__(
        self,
        runner: AgentRunner,
        search_tool: BaseTool,
    ) -> None:
        self._runner = runner
        self._search_tool = search_tool

    def run(self, query: str, response_language: str = "uk") -> ResearchResult:
        """Run one independent research request."""

        normalized_query = query.strip()
        normalized_language = response_language.strip().lower()

        if not normalized_query:
            raise ValueError("Research query must not be empty")

        if not normalized_language:
            raise ValueError("Response language must not be empty")

        tool_call = _create_initial_search_call(normalized_query)
        search_message = self._search_tool.invoke(tool_call)

        if not isinstance(search_message, ToolMessage):
            raise InvalidResearchResultError("Initial search did not return a ToolMessage")

        _validate_initial_search_message(search_message)

        state = self._runner.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"User request:\n{normalized_query}\n\n"
                            f"Response language ISO code: {normalized_language}\n\n"
                            "Important output requirement: return only the answer body. "
                            "Do not include URLs, Markdown links, or a sources section. "
                            "Verified sources are rendered separately by the application."
                        )
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[tool_call],
                    ),
                    search_message,
                ]
            }
        )

        messages = _validate_messages(state.get("messages"))
        answer = _sanitize_answer(_extract_final_answer(messages))
        sources = _extract_sources(messages)

        return ResearchResult(
            answer=answer,
            sources=sources,
        )


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
        source_artifacts: list[dict[str, object]] = []

        for index, item in enumerate(results, start=1):
            content_parts.append(
                "\n".join(
                    [
                        f"[{index}] {item.source.title}",
                        f"URL: {item.source.url}",
                        f"Source type: {item.source.source_type}",
                        f"Relevance score: {item.score:.3f}",
                        f"Content: {item.content}",
                    ]
                )
            )
            source_artifacts.append(item.source.model_dump(mode="json"))

        return "\n\n".join(content_parts), source_artifacts

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


def create_research_agent(
    search_service: SearchService | None = None,
    model: BaseChatModel | None = None,
) -> ResearchAgent:
    """Create the Ollama-backed Research Agent."""

    resolved_search_service = search_service or create_tavily_search_service()
    resolved_model = model or create_chat_model()
    search_tool = create_web_search_tool(resolved_search_service)

    runner = cast(
        AgentRunner,
        create_agent(
            model=resolved_model,
            tools=[search_tool],
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            name="research_agent",
        ),
    )

    return ResearchAgent(
        runner=runner,
        search_tool=search_tool,
    )


def _create_initial_search_call(query: str) -> ToolCall:
    search_query = _build_initial_search_query(query)

    return ToolCall(
        name="search_web",
        args={
            "query": search_query,
            "max_results": 5,
        },
        id=f"initial-search-{uuid4()}",
        type="tool_call",
    )


def _build_initial_search_query(query: str) -> str:
    search_focus = query

    if not query.isascii():
        technical_terms: list[str] = []
        seen_terms: set[str] = set()

        for match in TECHNICAL_TERM_PATTERN.finditer(query):
            term = match.group(0)
            term_key = term.casefold()

            if term_key in seen_terms:
                continue

            seen_terms.add(term_key)
            technical_terms.append(term)

        if technical_terms:
            search_focus = " ".join(technical_terms)

    return f"{search_focus} {OFFICIAL_SOURCE_HINT}"[:500]


def _validate_initial_search_message(message: ToolMessage) -> None:
    if message.status == "error":
        if isinstance(message.content, str):
            detail = message.content
        else:
            detail = "Initial web search failed"

        raise ResearchSearchError(detail)

    artifact = message.artifact

    if not isinstance(artifact, list) or not artifact:
        raise ResearchSearchError("Initial web search returned no verified sources")


def _validate_messages(value: object) -> list[BaseMessage]:
    if not isinstance(value, list):
        raise InvalidResearchResultError("Research agent returned no message list")

    messages: list[BaseMessage] = []

    for item in value:
        if not isinstance(item, BaseMessage):
            raise InvalidResearchResultError("Research agent returned an invalid message")
        messages.append(item)

    return messages


def _extract_final_answer(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            answer = message.content.strip()

            if answer:
                return answer

    raise InvalidResearchResultError("Research agent returned no final text answer")


def _extract_sources(messages: list[BaseMessage]) -> list[SourceItem]:
    sources: list[SourceItem] = []
    seen_urls: set[str] = set()

    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != "search_web":
            continue

        artifact = message.artifact

        if artifact is None:
            continue

        if not isinstance(artifact, list):
            raise InvalidResearchResultError("Search tool returned an invalid source artifact")

        for raw_source in artifact:
            try:
                source = SourceItem.model_validate(raw_source)
            except ValidationError as exc:
                raise InvalidResearchResultError(
                    "Search tool returned invalid source metadata"
                ) from exc

            url_key = str(source.url)

            if url_key in seen_urls:
                continue

            seen_urls.add(url_key)
            sources.append(source)

    return sources


def _sanitize_answer(answer: str) -> str:
    sanitized_lines: list[str] = []

    for line in answer.splitlines():
        if SOURCE_HEADING_PATTERN.fullmatch(line):
            continue

        if URL_PATTERN.search(line) and URL_LIST_ITEM_PATTERN.match(line):
            continue

        sanitized_lines.append(URL_PATTERN.sub("", line).rstrip())

    while sanitized_lines and not sanitized_lines[-1].strip():
        sanitized_lines.pop()

    sanitized = re.sub(r"\n{3,}", "\n\n", "\n".join(sanitized_lines)).strip()

    if not sanitized:
        raise InvalidResearchResultError(
            "Research agent returned no usable text after URL sanitization"
        )

    return sanitized
