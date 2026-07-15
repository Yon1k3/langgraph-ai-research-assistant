import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ai_research_assistant.agents.code import (
    CodeAgent,
    _build_code_search_queries,
    _build_repository_queries,
    _coerce_code_synthesis,
    _evidence_supports_request,
    _get_code_validation_feedback,
    _select_repository_candidate,
)
from ai_research_assistant.errors import CodeEvidenceError, InvalidCodeResultError
from ai_research_assistant.models import (
    CodeSynthesis,
    EvidenceItem,
    SourceItem,
    build_source_id,
)


class _InitialSearchInput(BaseModel):
    query: str = Field(min_length=1)
    max_results: int = Field(ge=1, le=10)

    model_config = ConfigDict(extra="forbid")


class FakeRunner:
    def __init__(self, additional_messages: list[BaseMessage] | None = None) -> None:
        self.additional_messages = additional_messages or []
        self.inputs: list[dict[str, object]] = []

    def invoke(
        self,
        input: dict[str, object],
        config: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.inputs.append(input)
        assert config == {"recursion_limit": 6}
        messages = input["messages"]
        assert isinstance(messages, list)
        return {
            "messages": [
                *messages,
                *self.additional_messages,
                AIMessage(content="Evidence collection complete."),
            ]
        }


class FakeSynthesizer:
    def __init__(self, result: CodeSynthesis) -> None:
        self.result = result
        self.calls: list[tuple[str, str, list[EvidenceItem]]] = []

    def __call__(
        self,
        query: str,
        language: str,
        evidence: list[EvidenceItem],
    ) -> CodeSynthesis:
        self.calls.append((query, language, evidence))
        return self.result


def _make_evidence(
    *,
    title: str,
    url: str,
    source_type: str = "documentation",
    content: str = (
        "LangGraph exports START and StateGraph. "
        "StateGraph accepts a state schema and nodes before compilation."
    ),
) -> EvidenceItem:
    source = SourceItem(
        title=title,
        url=url,
        source_type=source_type,
    )
    return EvidenceItem(
        source_id=build_source_id(url),
        source=source,
        content=content,
        score=1.0,
    )


def _make_initial_tool(
    evidence: list[EvidenceItem],
    calls: list[tuple[str, int]],
) -> BaseTool:
    def search_github_repositories(
        query: str,
        max_results: int,
    ) -> tuple[str, list[dict[str, object]]]:
        calls.append((query, max_results))
        return (
            "Repository search results",
            [item.model_dump(mode="json") for item in evidence],
        )

    return StructuredTool.from_function(
        func=search_github_repositories,
        name="search_github_repositories",
        description="Find public repositories.",
        args_schema=_InitialSearchInput,
        response_format="content_and_artifact",
    )


def _make_agent(
    synthesis: CodeSynthesis,
    *,
    initial_evidence: list[EvidenceItem] | None = None,
    additional_messages: list[BaseMessage] | None = None,
) -> tuple[CodeAgent, FakeRunner, FakeSynthesizer, list[tuple[str, int]]]:
    repository_evidence = initial_evidence or [
        _make_evidence(
            title="langchain-ai/langgraph",
            url="https://github.com/langchain-ai/langgraph",
            source_type="github",
            content="Official LangGraph repository metadata.",
        )
    ]
    calls: list[tuple[str, int]] = []
    runner = FakeRunner(additional_messages)
    synthesizer = FakeSynthesizer(synthesis)
    agent = CodeAgent(
        runner=runner,
        initial_search_tool=_make_initial_tool(repository_evidence, calls),
        synthesize=synthesizer,
    )
    return agent, runner, synthesizer, calls


def test_code_agent_returns_fenced_code_and_only_selected_sources() -> None:
    documentation = _make_evidence(
        title="Use the graph API",
        url="https://docs.langchain.com/oss/python/langgraph/use-graph-api",
    )
    additional_message = ToolMessage(
        content="Official documentation result",
        tool_call_id="documentation-call",
        name="search_official_documentation",
        artifact=[documentation.model_dump(mode="json")],
    )
    synthesis = CodeSynthesis(
        explanation=(
            "Створи граф, додай вузол і скомпілюй його. "
            "Деталі: https://should-not-appear.example/docs"
        ),
        code=(
            "from langgraph.graph import START, StateGraph\n\n"
            "builder = StateGraph(dict)\n"
            'builder.add_node("answer", lambda state: state)\n'
            'builder.add_edge(START, "answer")\n'
            "graph = builder.compile()"
        ),
        code_language="python",
        used_source_ids=[documentation.source_id],
        is_sufficient=True,
    )
    agent, runner, synthesizer, calls = _make_agent(
        synthesis,
        additional_messages=[additional_message],
    )

    result = agent.run(
        "Покажи приклад StateGraph у LangGraph на Python.",
        response_language="uk",
    )

    assert calls == [("StateGraph", 5), ("LangGraph", 5)]
    assert len(runner.inputs) == 1
    assert len(synthesizer.calls) == 1
    assert "https://" not in result.answer
    assert "```python" in result.answer
    assert "builder.compile()" in result.answer
    assert [source.source_id for source in result.sources] == [documentation.source_id]
    assert result.claims == []


def test_code_agent_accepts_grounded_import_only_example() -> None:
    repository = _make_evidence(
        title="langchain-ai/langgraph",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
        content="Official LangGraph repository metadata without import examples.",
    )
    documentation = _make_evidence(
        title="StateGraph and START constants",
        url="https://reference.langchain.com/python/langgraph/graph/",
        content="from langgraph.graph import START, StateGraph",
    )
    unrelated_documentation = _make_evidence(
        title="interrupt | langgraph | LangChain Reference",
        url="https://reference.langchain.com/python/langgraph/types/interrupt",
        content=("Navigation text containing from langgraph.graph import START, StateGraph"),
    )
    synthesis = CodeSynthesis(
        explanation="Імпортуй класи з офіційного модуля графів.",
        code="from langgraph.graph import START, StateGraph",
        code_language="python",
        used_source_ids=[repository.source_id],
        is_sufficient=True,
    )
    agent, _, _, _ = _make_agent(
        synthesis,
        initial_evidence=[repository],
        additional_messages=[
            ToolMessage(
                content="Official reference",
                tool_call_id="documentation-call",
                name="search_official_documentation",
                artifact=[
                    unrelated_documentation.model_dump(mode="json"),
                    documentation.model_dump(mode="json"),
                ],
            )
        ],
    )

    result = agent.run(
        "Покажи правильний Python-імпорт StateGraph і START з LangGraph.",
        response_language="uk",
    )

    assert "from langgraph.graph import START, StateGraph" in result.answer
    assert [source.source_id for source in result.sources] == [documentation.source_id]


def test_code_agent_returns_localized_fallback_for_insufficient_evidence() -> None:
    agent, _, _, _ = _make_agent(CodeSynthesis(is_sufficient=False))

    result = agent.run("Show a library example.", response_language="uk")

    assert "Не вдалося" in result.answer
    assert result.sources == []


def test_code_agent_rejects_unknown_synthesis_source_id() -> None:
    synthesis = CodeSynthesis(
        explanation="Use the documented API.",
        code="print('safe')",
        code_language="python",
        used_source_ids=[build_source_id("https://example.com/invented")],
        is_sufficient=True,
    )
    agent, _, _, _ = _make_agent(synthesis)

    result = agent.run("Show a LangGraph example.", response_language="en")

    assert result.answer.startswith("A reliable code example could not")
    assert result.sources == []


def test_code_agent_rejects_imports_missing_from_selected_evidence() -> None:
    repository = _make_evidence(
        title="langchain-ai/langgraph",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
        content="Official LangGraph repository metadata without this invented API.",
    )
    synthesis = CodeSynthesis(
        explanation="Use the generated integration.",
        code=("from langchain.tools.stategraph import StateGraphTool\n\ntool = StateGraphTool()"),
        code_language="python",
        used_source_ids=[repository.source_id],
        is_sufficient=True,
    )
    agent, _, _, _ = _make_agent(
        synthesis,
        initial_evidence=[repository],
    )

    result = agent.run("Show a LangGraph StateGraph example.", response_language="en")

    assert result.answer.startswith("A reliable code example could not")
    assert result.sources == []


def test_code_agent_rejects_code_that_does_not_satisfy_requested_actions() -> None:
    repository = _make_evidence(
        title="langchain-ai/langgraph",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
        content=(
            "Stars: 25000\n"
            "from langgraph.graph import START, StateGraph\n"
            "builder.add_node; builder.add_edge; builder.compile"
        ),
    )
    synthesis = CodeSynthesis(
        explanation="Create and compile the graph.",
        code=(
            "from langgraph.graph import StateGraph\n\n"
            "builder = StateGraph(dict)\n"
            'builder.add_node("answer", lambda state: state)\n'
            "builder.compile()  # START and add_edge are only mentioned here"
        ),
        code_language="python",
        used_source_ids=[repository.source_id],
        is_sufficient=True,
    )
    agent, _, _, _ = _make_agent(
        synthesis,
        initial_evidence=[repository],
    )

    result = agent.run(
        "Show one node, an edge from START, and compile the LangGraph graph.",
        response_language="en",
    )

    assert result.answer.startswith("A reliable code example could not")


def test_code_agent_rejects_undefined_python_names() -> None:
    documentation = _make_evidence(
        title="langchain-ai/langgraph",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
        content=(
            "from langgraph.graph import START, StateGraph\n"
            "builder = StateGraph(State)\n"
            "builder.add_node('a', node_a)\n"
            "builder.add_edge(START, 'a')\n"
            "graph = builder.compile()"
        ),
    )
    synthesis = CodeSynthesis(
        explanation="Build and compile the graph.",
        code=(
            "from langgraph.graph import START, StateGraph\n\n"
            "builder = StateGraph(State)\n"
            'builder.add_node("a", node_a)\n'
            'builder.add_edge(START, "a")\n'
            "graph = builder.compile()"
        ),
        code_language="python",
        used_source_ids=[documentation.source_id],
        is_sufficient=True,
    )
    agent, _, _, _ = _make_agent(
        synthesis,
        initial_evidence=[documentation],
    )

    result = agent.run(
        "Show one node, an edge from START, and compile the LangGraph graph.",
        response_language="en",
    )

    assert result.answer.startswith("A reliable code example could not")

    feedback = _get_code_validation_feedback(
        "Show one node, an edge from START, and compile the LangGraph graph.",
        synthesis,
        [documentation],
    )

    assert any("State, node_a" in item for item in feedback)


def test_code_agent_fails_when_initial_search_returns_no_evidence() -> None:
    agent, _, _, _ = _make_agent(
        CodeSynthesis(is_sufficient=False),
        initial_evidence=[],
    )
    calls: list[tuple[str, int]] = []
    agent = CodeAgent(
        runner=FakeRunner(),
        initial_search_tool=_make_initial_tool([], calls),
        synthesize=FakeSynthesizer(CodeSynthesis(is_sufficient=False)),
    )

    with pytest.raises(CodeEvidenceError, match="returned no evidence"):
        agent.run("Show a LangGraph example.", response_language="en")


def test_code_agent_rejects_invalid_tool_artifact() -> None:
    invalid_message = ToolMessage(
        content="Invalid evidence",
        tool_call_id="invalid-call",
        name="read_github_readme",
        artifact=[{"unexpected": "data"}],
    )
    agent, _, _, _ = _make_agent(
        CodeSynthesis(is_sufficient=False),
        additional_messages=[invalid_message],
    )

    with pytest.raises(InvalidCodeResultError, match="invalid evidence data"):
        agent.run("Show a LangGraph example.", response_language="en")


@pytest.mark.parametrize("query", ["", "   "])
def test_code_agent_rejects_empty_query(query: str) -> None:
    agent, _, _, _ = _make_agent(CodeSynthesis(is_sufficient=False))

    with pytest.raises(ValueError, match="must not be empty"):
        agent.run(query)


def test_code_synthesis_coercion_supports_wrapped_tool_arguments() -> None:
    source_id = build_source_id("https://docs.example.com/api")
    result = _coerce_code_synthesis(
        {
            "raw": AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "CodeSynthesis",
                        "args": {
                            "explanation": "Use the API.",
                            "code": "print('ok')",
                            "code_language": "python",
                            "used_source_ids": [source_id],
                            "is_sufficient": True,
                        },
                        "id": "structured-call",
                        "type": "tool_call",
                    }
                ],
            )
        }
    )

    assert result.is_sufficient is True
    assert result.used_source_ids == [source_id]


def test_repository_query_uses_technical_names_not_generic_terms() -> None:
    assert _build_repository_queries("Write a Python integration example for FastAPI") == [
        "FastAPI"
    ]


def test_code_search_queries_include_requested_graph_operations() -> None:
    assert _build_code_search_queries(
        "Show one StateGraph node, an edge from START, and compile LangGraph.",
        "langgraph",
    ) == ["StateGraph", "add_node", "add_edge", "compile", "START"]


def test_evidence_readiness_requires_every_requested_operation() -> None:
    evidence = _make_evidence(
        title="langchain-ai/langgraph",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
        content="LangGraph StateGraph add_node add_edge compile START",
    )
    query = "Show one StateGraph node, an edge from START, and compile LangGraph."

    assert _evidence_supports_request(query, [evidence]) is True
    assert (
        _evidence_supports_request(
            query,
            [evidence.model_copy(update={"content": "LangGraph StateGraph add_node"})],
        )
        is False
    )


def test_repository_selection_prefers_established_exact_name_match() -> None:
    state_graph = _make_evidence(
        title="example/StateGraph",
        url="https://github.com/example/StateGraph",
        source_type="github",
        content="Stars: 12",
    )
    langgraph = _make_evidence(
        title="langchain-ai/langgraph",
        url="https://github.com/langchain-ai/langgraph",
        source_type="github",
        content="Stars: 25000",
    )
    state_graph_call = {
        "name": "search_github_repositories",
        "args": {"query": "StateGraph", "max_results": 5},
        "id": "state-graph-call",
        "type": "tool_call",
    }
    langgraph_call = {
        "name": "search_github_repositories",
        "args": {"query": "LangGraph", "max_results": 5},
        "id": "langgraph-call",
        "type": "tool_call",
    }
    state_graph_message = ToolMessage(
        content="StateGraph",
        tool_call_id="state-graph-call",
        artifact=[state_graph.model_dump(mode="json")],
    )
    langgraph_message = ToolMessage(
        content="LangGraph",
        tool_call_id="langgraph-call",
        artifact=[langgraph.model_dump(mode="json")],
    )

    selected = _select_repository_candidate(
        [
            ("StateGraph", state_graph_call, state_graph_message, state_graph),
            ("LangGraph", langgraph_call, langgraph_message, langgraph),
        ]
    )

    assert selected is not None
    assert selected[3].source.title == "langchain-ai/langgraph"


def test_code_agent_has_no_execution_callable() -> None:
    agent, _, _, _ = _make_agent(CodeSynthesis(is_sufficient=False))

    forbidden_attributes = {"execute", "run_python", "python_repl", "shell"}

    assert not any(hasattr(agent, attribute) for attribute in forbidden_attributes)
