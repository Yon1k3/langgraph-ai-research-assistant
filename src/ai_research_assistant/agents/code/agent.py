from langchain_core.tools import BaseTool

from ai_research_assistant.agents.code.constants import CODE_AGENT_RECURSION_LIMIT
from ai_research_assistant.agents.code.evidence import (
    _collect_initial_source_messages,
    _extract_evidence,
    _prepare_evidence,
    _resolve_used_sources,
    _validate_messages,
)
from ai_research_assistant.agents.code.synthesis import _create_insufficient_result
from ai_research_assistant.agents.code.types import AgentRunner, CodeSynthesizer
from ai_research_assistant.agents.code.validation import (
    _attach_supporting_sources,
    _is_code_supported_by_evidence,
    _sanitize_explanation,
)
from ai_research_assistant.errors import CodeEvidenceError, InvalidCodeResultError
from ai_research_assistant.models import CodeResult, CodeSynthesis


class CodeAgent:
    """Generate grounded code examples without executing user or model code."""

    def __init__(
        self,
        runner: AgentRunner,
        initial_search_tool: BaseTool,
        synthesize: CodeSynthesizer,
        documentation_tool: BaseTool | None = None,
        readme_tool: BaseTool | None = None,
        code_search_tool: BaseTool | None = None,
    ) -> None:
        self._runner = runner
        self._initial_search_tool = initial_search_tool
        self._synthesize = synthesize
        self._documentation_tool = documentation_tool
        self._readme_tool = readme_tool
        self._code_search_tool = code_search_tool

    def run(
        self,
        query: str,
        response_language: str = "uk",
    ) -> CodeResult:
        """Run one independent code-generation request without execution."""

        normalized_query = query.strip()
        normalized_language = response_language.strip().lower()

        if not normalized_query:
            raise ValueError("Code query must not be empty")

        if not normalized_language:
            raise ValueError("Response language must not be empty")

        source_messages = _collect_initial_source_messages(
            normalized_query,
            self._initial_search_tool,
            self._documentation_tool,
            self._readme_tool,
            self._code_search_tool,
        )

        state = self._runner.invoke(
            {
                "messages": source_messages,
            },
            {"recursion_limit": CODE_AGENT_RECURSION_LIMIT},
        )

        messages = _validate_messages(state.get("messages"))
        evidence = _prepare_evidence(
            _extract_evidence(messages),
            normalized_query,
        )

        if not evidence:
            raise CodeEvidenceError("Code Agent collected no usable source evidence")

        synthesis = self._synthesize(
            normalized_query,
            normalized_language,
            evidence,
        )

        if not isinstance(synthesis, CodeSynthesis):
            raise InvalidCodeResultError("Code synthesizer returned an unexpected response type")

        if not synthesis.is_sufficient:
            return _create_insufficient_result(normalized_language)

        synthesis = _attach_supporting_sources(
            normalized_query,
            synthesis,
            evidence,
        )

        if not _is_code_supported_by_evidence(
            normalized_query,
            synthesis,
            evidence,
        ):
            return _create_insufficient_result(normalized_language)

        sources = _resolve_used_sources(synthesis.used_source_ids, evidence)

        if not sources:
            return _create_insufficient_result(normalized_language)

        explanation = _sanitize_explanation(synthesis.explanation)
        answer = f"{explanation}\n\n```{synthesis.code_language}\n{synthesis.code.rstrip()}\n```"

        return CodeResult(answer=answer, sources=sources)
