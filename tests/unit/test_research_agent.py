import hashlib
import json

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from ai_research_assistant.agents import (
    InvalidResearchResultError,
    ResearchAgent,
    ResearchSearchError,
    create_web_search_tool,
)
from ai_research_assistant.agents.research import _coerce_research_synthesis
from ai_research_assistant.models import (
    EvidenceItem,
    GroundedClaim,
    ResearchSynthesis,
    SearchResultItem,
    SourceItem,
)
from ai_research_assistant.tools.web_search import SearchUnavailableError


class FakeSearchService:
    def __init__(self, results: list[SearchResultItem]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]:
        self.calls.append((query, max_results))
        return self.results


class FailingSearchService:
    def search(self, query: str, max_results: int = 5) -> list[SearchResultItem]:
        raise SearchUnavailableError("Search is unavailable")


class FakeRunner:
    def __init__(self, response_messages: list[BaseMessage] | None = None) -> None:
        self.response_messages = response_messages or []
        self.last_input: dict[str, object] | None = None

    def invoke(self, input: dict[str, object]) -> dict[str, object]:
        self.last_input = input
        input_messages = input.get("messages")

        if not isinstance(input_messages, list):
            raise AssertionError("Runner received no message list")

        return {
            "messages": [
                *input_messages,
                *self.response_messages,
            ]
        }


class FakeSynthesizer:
    def __init__(
        self,
        *,
        answer: str = "LangGraph grounded answer",
        used_indexes: tuple[int, ...] = (0,),
        explicit_source_ids: list[str] | None = None,
        supporting_quote: str | None = None,
        is_sufficient: bool = True,
    ) -> None:
        self.answer = answer
        self.used_indexes = used_indexes
        self.explicit_source_ids = explicit_source_ids
        self.supporting_quote = supporting_quote
        self.is_sufficient = is_sufficient
        self.calls: list[tuple[str, str, list[EvidenceItem]]] = []

    def __call__(
        self,
        query: str,
        response_language: str,
        evidence: list[EvidenceItem],
    ) -> ResearchSynthesis:
        self.calls.append((query, response_language, evidence))

        if not self.is_sufficient:
            return ResearchSynthesis(
                claims=[],
                is_sufficient=False,
            )

        source_ids = self.explicit_source_ids or [
            evidence[index].source_id for index in self.used_indexes
        ]

        claims = []

        for position, source_id in enumerate(source_ids):
            evidence_index = self.used_indexes[position] if position < len(self.used_indexes) else 0
            claims.append(
                GroundedClaim(
                    statement=self.answer,
                    source_id=source_id,
                    supporting_quote=(self.supporting_quote or evidence[evidence_index].content),
                )
            )

        return ResearchSynthesis(claims=claims, is_sufficient=True)


def make_source(
    *,
    title: str = "LangGraph overview",
    url: str = "https://docs.langchain.com/oss/python/langgraph/overview",
    source_type: str = "documentation",
) -> SourceItem:
    return SourceItem(
        title=title,
        url=url,
        source_type=source_type,
    )


def make_search_result(
    source: SourceItem,
    *,
    content: str = "LangGraph is a low-level orchestration framework.",
    score: float = 0.95,
) -> SearchResultItem:
    return SearchResultItem(
        source=source,
        content=content,
        score=score,
    )


def make_evidence(source: SourceItem) -> EvidenceItem:
    source_digest = hashlib.sha256(str(source.url).encode("utf-8")).hexdigest()[:12]

    return EvidenceItem(
        source_id=f"src-{source_digest}",
        source=source,
        content="Additional search evidence.",
        score=0.9,
    )


def make_research_agent(
    runner: FakeRunner,
    service: FakeSearchService | None = None,
    synthesizer: FakeSynthesizer | None = None,
) -> tuple[ResearchAgent, FakeSearchService, FakeSynthesizer]:
    resolved_service = service or FakeSearchService([make_search_result(make_source())])
    resolved_synthesizer = synthesizer or FakeSynthesizer()
    search_tool = create_web_search_tool(resolved_service)

    return (
        ResearchAgent(
            runner=runner,
            search_tool=search_tool,
            synthesize=resolved_synthesizer,
        ),
        resolved_service,
        resolved_synthesizer,
    )


def test_search_tool_returns_stable_evidence_artifact() -> None:
    source = make_source()
    service = FakeSearchService([make_search_result(source)])
    search_tool = create_web_search_tool(service)
    tool_call = {
        "name": "search_web",
        "args": {
            "query": "LangGraph official documentation",
            "max_results": 3,
        },
        "id": "call-1",
        "type": "tool_call",
    }

    first_message = search_tool.invoke(tool_call)
    second_message = search_tool.invoke({**tool_call, "id": "call-2"})

    assert isinstance(first_message, ToolMessage)
    assert isinstance(second_message, ToolMessage)
    assert isinstance(first_message.artifact, list)
    assert isinstance(second_message.artifact, list)

    first_evidence = EvidenceItem.model_validate(first_message.artifact[0])
    second_evidence = EvidenceItem.model_validate(second_message.artifact[0])

    assert first_evidence.source_id == second_evidence.source_id
    assert first_evidence.source == source
    assert "LangGraph is a low-level orchestration framework." in first_message.content
    assert first_evidence.source_id in first_message.content
    assert service.calls == [
        ("LangGraph official documentation", 3),
        ("LangGraph official documentation", 3),
    ]


def test_synthesis_parser_uses_parsed_pydantic_result() -> None:
    expected = ResearchSynthesis(
        claims=[
            GroundedClaim(
                statement="LangGraph supports stateful workflows.",
                source_id="src-0123456789ab",
                supporting_quote="LangGraph supports stateful agent workflows.",
            )
        ],
        is_sufficient=True,
    )

    result = _coerce_research_synthesis({"parsed": expected})

    assert result == expected


def test_synthesis_parser_recovers_textual_function_call() -> None:
    payload = {
        "claims": [
            {
                "statement": "LangGraph підтримує агентні робочі процеси зі станом.",
                "source_id": "src-0123456789ab",
                "supporting_quote": "LangGraph supports stateful agent workflows.",
            }
        ],
        "is_sufficient": True,
    }
    malformed_wrapper = '{"name":"ResearchSynthesis","parameters":' + json.dumps(payload)

    result = _coerce_research_synthesis(
        {
            "parsed": None,
            "raw": AIMessage(content=malformed_wrapper),
        }
    )

    assert result.is_sufficient is True
    assert result.claims[0].source_id == "src-0123456789ab"


def test_synthesis_parser_recovers_raw_tool_call_arguments() -> None:
    payload = {
        "claims": [
            {
                "statement": "LangGraph supports stateful workflows.",
                "source_id": "src-0123456789ab",
                "supporting_quote": "LangGraph supports stateful agent workflows.",
            }
        ],
        "is_sufficient": True,
    }
    raw_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ResearchSynthesis",
                "args": payload,
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    result = _coerce_research_synthesis(
        {
            "parsed": None,
            "raw": raw_message,
        }
    )

    assert result.is_sufficient is True
    assert result.claims[0].statement == "LangGraph supports stateful workflows."


def test_synthesis_parser_falls_back_for_invalid_model_output() -> None:
    result = _coerce_research_synthesis(
        {
            "parsed": None,
            "raw": AIMessage(content="Evidence collection is complete."),
        }
    )

    assert result.is_sufficient is False
    assert result.claims == []


def test_research_agent_stops_when_initial_search_fails() -> None:
    runner = FakeRunner()
    search_tool = create_web_search_tool(FailingSearchService())
    synthesizer = FakeSynthesizer()
    agent = ResearchAgent(
        runner=runner,
        search_tool=search_tool,
        synthesize=synthesizer,
    )

    with pytest.raises(
        ResearchSearchError,
        match="Search is unavailable",
    ):
        agent.run("Explain LangGraph")

    assert runner.last_input is None
    assert synthesizer.calls == []


def test_research_agent_stops_when_initial_search_has_no_sources() -> None:
    runner = FakeRunner()
    agent, _, synthesizer = make_research_agent(
        runner,
        service=FakeSearchService([]),
    )

    with pytest.raises(
        ResearchSearchError,
        match="no verified sources",
    ):
        agent.run("Explain LangGraph")

    assert runner.last_input is None
    assert synthesizer.calls == []


def test_search_tool_converts_provider_error_to_tool_error() -> None:
    search_tool = create_web_search_tool(FailingSearchService())

    message = search_tool.invoke(
        {
            "name": "search_web",
            "args": {"query": "LangGraph"},
            "id": "call-1",
            "type": "tool_call",
        }
    )

    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "Search is unavailable" in message.content


def test_research_agent_returns_only_sources_selected_by_synthesis() -> None:
    documentation = make_source()
    repository = make_source(
        title="LangGraph repository",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
    )
    service = FakeSearchService(
        [
            make_search_result(documentation),
            make_search_result(repository, score=0.9),
        ]
    )
    runner = FakeRunner()
    synthesizer = FakeSynthesizer(
        answer="LangGraph supports stateful agent workflows.",
        used_indexes=(1,),
    )
    agent, _, _ = make_research_agent(runner, service, synthesizer)

    result = agent.run("Explain LangGraph", response_language="en")

    assert result.answer == "LangGraph supports stateful agent workflows."
    assert result.sources == [repository]
    assert len(synthesizer.calls) == 1

    query, language, evidence = synthesizer.calls[0]
    assert query == "Explain LangGraph"
    assert language == "en"
    assert [item.source for item in evidence] == [documentation, repository]


def test_research_agent_focuses_non_english_search_on_technical_terms() -> None:
    runner = FakeRunner()
    agent, service, _ = make_research_agent(runner)

    agent.run(
        "Поясни актуальне призначення LangGraph та наведи офіційні джерела.",
        response_language="uk",
    )

    search_query, _ = service.calls[0]

    assert search_query.startswith("LangGraph official documentation")
    assert "офіційні джерела" not in search_query


def test_research_agent_targets_release_sources_for_version_request() -> None:
    agent, service, _ = make_research_agent(FakeRunner())

    agent.run("Яка остання версія LangGraph?", response_language="uk")

    search_query, _ = service.calls[0]
    assert search_query.startswith("LangGraph official release notes changelog")


def test_research_agent_deduplicates_evidence_from_multiple_searches() -> None:
    source = make_source()
    evidence = make_evidence(source)
    runner = FakeRunner(
        [
            ToolMessage(
                content="Additional search result",
                tool_call_id="call-2",
                name="search_web",
                artifact=[evidence.model_dump(mode="json")],
            )
        ]
    )
    agent, _, synthesizer = make_research_agent(runner)

    result = agent.run("Explain LangGraph", response_language="en")

    assert result.sources == [source]
    assert len(synthesizer.calls[0][2]) == 1


def test_research_agent_prioritizes_official_evidence_for_synthesis() -> None:
    web_source = make_source(
        title="Community article",
        url="https://example.com/langgraph",
        source_type="web",
    )
    release_source = make_source(
        title="LangGraph releases",
        url="https://github.com/langchain-ai/langgraph/releases",
        source_type="release_notes",
    )
    repository_source = make_source(
        title="LangGraph repository",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
    )
    documentation_source = make_source()
    service = FakeSearchService(
        [
            make_search_result(web_source, score=0.99),
            make_search_result(release_source, score=0.7),
            make_search_result(repository_source, score=0.8),
            make_search_result(documentation_source, score=0.6),
        ]
    )
    synthesizer = FakeSynthesizer()
    agent, _, _ = make_research_agent(FakeRunner(), service, synthesizer)

    agent.run("Explain LangGraph")

    evidence = synthesizer.calls[0][2]
    assert [item.source.source_type for item in evidence] == [
        "documentation",
        "github",
        "release_notes",
    ]


def test_research_agent_limits_evidence_sent_to_synthesis() -> None:
    source_types = [
        "documentation",
        "github",
        "release_notes",
        "documentation",
        "github",
        "release_notes",
    ]
    results = [
        make_search_result(
            make_source(
                title=f"Documentation page {index}",
                url=f"https://docs.example.com/page-{index}",
                source_type=source_types[index],
            ),
            content="C" * 2_000,
            score=0.9 - index * 0.01,
        )
        for index in range(6)
    ]
    synthesizer = FakeSynthesizer()
    agent, _, _ = make_research_agent(
        FakeRunner(),
        FakeSearchService(results),
        synthesizer,
    )

    agent.run("Explain the documented feature")

    evidence = synthesizer.calls[0][2]
    assert len(evidence) == 3
    assert all(len(item.content) == 1_000 for item in evidence)


def test_research_agent_selects_factual_excerpt_instead_of_navigation() -> None:
    navigation = " ".join(
        f"* [Navigation item {index}](https://docs.example.com/nav-{index})." for index in range(40)
    )
    factual_content = (
        "LangGraph is an orchestration runtime focused on durable execution, "
        "streaming, human-in-the-loop, and persistence."
    )
    service = FakeSearchService(
        [
            make_search_result(
                make_source(),
                content=f"{navigation} {factual_content}",
            )
        ]
    )
    synthesizer = FakeSynthesizer()
    agent, _, _ = make_research_agent(FakeRunner(), service, synthesizer)

    agent.run("Explain LangGraph")

    excerpt = synthesizer.calls[0][2][0].content
    assert factual_content in excerpt
    assert "Navigation item" not in excerpt


def test_research_agent_filters_evidence_unrelated_to_subject() -> None:
    unrelated_source = make_source(
        title="LangSmith changelog",
        url="https://docs.langchain.com/langsmith/changelog",
        source_type="release_notes",
    )
    relevant_source = make_source()
    service = FakeSearchService(
        [
            make_search_result(
                unrelated_source,
                content="LangSmith observability release updates.",
            ),
            make_search_result(relevant_source),
        ]
    )
    synthesizer = FakeSynthesizer()
    agent, _, _ = make_research_agent(FakeRunner(), service, synthesizer)

    result = agent.run("Explain LangGraph")

    evidence = synthesizer.calls[0][2]
    assert [item.source for item in evidence] == [relevant_source]
    assert result.sources == [relevant_source]


def test_research_agent_rejects_invalid_evidence_artifact() -> None:
    runner = FakeRunner(
        [
            ToolMessage(
                content="Search result",
                tool_call_id="call-2",
                name="search_web",
                artifact=[{"invalid": "evidence"}],
            )
        ]
    )
    agent, _, _ = make_research_agent(runner)

    with pytest.raises(InvalidResearchResultError, match="invalid evidence metadata"):
        agent.run("Explain LangGraph")


def test_research_agent_ignores_unknown_synthesis_source_id() -> None:
    synthesizer = FakeSynthesizer(
        explicit_source_ids=["src-ffffffffffff"],
    )
    agent, _, _ = make_research_agent(FakeRunner(), synthesizer=synthesizer)

    result = agent.run("Explain LangGraph", response_language="en")

    assert result.answer.startswith("A reliable answer could not be produced")
    assert result.sources == []


def test_research_agent_ignores_claim_without_exact_supporting_quote() -> None:
    synthesizer = FakeSynthesizer(
        supporting_quote="This quote does not exist in the selected evidence.",
    )
    agent, _, _ = make_research_agent(FakeRunner(), synthesizer=synthesizer)

    result = agent.run("Explain LangGraph", response_language="en")

    assert result.answer.startswith("A reliable answer could not be produced")
    assert result.sources == []


def test_research_agent_ignores_claim_about_another_technology() -> None:
    synthesizer = FakeSynthesizer(
        answer="LangChain supports complex stateful workflows.",
    )
    agent, _, _ = make_research_agent(FakeRunner(), synthesizer=synthesizer)

    result = agent.run("Explain LangGraph", response_language="en")

    assert result.answer.startswith("A reliable answer could not be produced")
    assert result.sources == []


def test_research_agent_returns_localized_fallback_for_insufficient_evidence() -> None:
    synthesizer = FakeSynthesizer(is_sufficient=False)
    agent, _, _ = make_research_agent(FakeRunner(), synthesizer=synthesizer)

    result = agent.run("Поясни невідому технологію", response_language="uk")

    assert result.answer.startswith("Не вдалося сформувати надійну відповідь")
    assert result.sources == []


def test_research_agent_rejects_empty_query() -> None:
    agent, _, _ = make_research_agent(FakeRunner())

    with pytest.raises(ValueError, match="must not be empty"):
        agent.run("   ")


def test_research_agent_removes_urls_from_synthesized_answer() -> None:
    synthesizer = FakeSynthesizer(
        answer=(
            "LangGraph supports stateful workflows.\n\n"
            "Sources:\n"
            "- LangGraph documentation: "
            "https://docs.langchain.com/oss/python/langgraph/overview"
        )
    )
    agent, _, _ = make_research_agent(FakeRunner(), synthesizer=synthesizer)

    result = agent.run("Explain LangGraph")

    assert result.answer == "LangGraph supports stateful workflows."
    assert "http" not in result.answer


def test_research_agent_falls_back_when_synthesized_answer_contains_only_urls() -> None:
    synthesizer = FakeSynthesizer(
        answer=(
            "Sources:\n"
            "- LangGraph documentation: "
            "https://docs.langchain.com/oss/python/langgraph/overview"
        )
    )
    agent, _, _ = make_research_agent(FakeRunner(), synthesizer=synthesizer)

    result = agent.run("Explain LangGraph", response_language="en")

    assert result.answer.startswith("A reliable answer could not be produced")
    assert result.sources == []
