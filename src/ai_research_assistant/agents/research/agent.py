from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from ai_research_assistant.agents.research.evidence import (
    _create_initial_search_call,
    _extract_evidence,
    _prepare_synthesis_evidence,
    _validate_initial_search_message,
    _validate_messages,
)
from ai_research_assistant.agents.research.types import AgentRunner, EvidenceSynthesizer
from ai_research_assistant.agents.research.validation import (
    _format_verified_claims,
    _get_insufficient_evidence_response,
    _resolve_used_sources,
    _select_verified_claims,
)
from ai_research_assistant.errors import (
    InvalidResearchResultError,
    ResearchSearchError,
)
from ai_research_assistant.models import ResearchResult, ResearchSynthesis


class ResearchAgent:
    """Run technical research and return an answer with verified sources."""

    def __init__(
        self,
        runner: AgentRunner,
        search_tool: BaseTool,
        synthesize: EvidenceSynthesizer,
    ) -> None:
        self._runner = runner
        self._search_tool = search_tool
        self._synthesize = synthesize

    def run(
        self,
        query: str,
        response_language: str = "uk",
    ) -> ResearchResult:
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
                            f"Collect evidence for this user request:\n{normalized_query}\n\n"
                            "Use the mandatory initial search result below. Search again "
                            "only if it does not directly address the request."
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
        evidence = _prepare_synthesis_evidence(
            _extract_evidence(messages),
            normalized_query,
        )

        if not evidence:
            raise ResearchSearchError("Research search returned no usable evidence")

        synthesis = self._synthesize(
            normalized_query,
            normalized_language,
            evidence,
        )

        if not isinstance(synthesis, ResearchSynthesis):
            raise InvalidResearchResultError(
                "Research synthesizer returned an unexpected response type"
            )

        verified_claims = _select_verified_claims(
            synthesis.claims,
            evidence,
            normalized_query,
        )

        if not synthesis.is_sufficient or not verified_claims:
            return ResearchResult(
                answer=_get_insufficient_evidence_response(normalized_language),
                sources=[],
            )

        answer = _format_verified_claims(verified_claims)
        used_source_ids = list(dict.fromkeys(claim.source_id for claim in verified_claims))
        sources = _resolve_used_sources(used_source_ids, evidence)

        return ResearchResult(
            answer=answer,
            sources=sources,
            claims=verified_claims,
        )
