import sqlite3
from pathlib import Path

import pytest
from langgraph.types import Command

from ai_research_assistant.conversation import ConversationContext
from ai_research_assistant.graph.builder import build_core_graph
from ai_research_assistant.graph.nodes import ResponseKind
from ai_research_assistant.memory import (
    create_sqlite_checkpointer,
    open_sqlite_checkpointer,
)
from ai_research_assistant.models import (
    AgentResult,
    CodeResult,
    ResearchResult,
    RouteDecision,
    SourceItem,
    SourceReference,
)


def test_checkpointer_preserves_messages_in_the_same_thread(tmp_path: Path) -> None:
    checkpointer = create_sqlite_checkpointer(tmp_path / "conversation.sqlite3")
    graph = build_core_graph(
        classify=_classify_direct_answer,
        generate=_generate_response,
        research=_unexpected_research,
        code=_unexpected_code,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "conversation-thread"}}

    graph.invoke(
        {"messages": [{"role": "user", "content": "First question"}]},
        config=config,
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Second question"}]},
        config=config,
    )

    assert [message.content for message in result["messages"]] == [
        "First question",
        "direct_answer:en:First question",
        "Second question",
        "direct_answer:en:Second question",
    ]

    checkpointer.conn.close()


def test_checkpointed_follow_up_uses_recent_conversation_context(tmp_path: Path) -> None:
    captured_follow_up_contexts: list[ConversationContext] = []
    captured_research_queries: list[str] = []

    def classify(context: ConversationContext) -> RouteDecision:
        if context.latest_user_query == "Поясни LangGraph.":
            return RouteDecision(
                route="direct_answer",
                confidence=1.0,
                reason="Initial technical question.",
                response_language="uk",
                resolved_query="Поясни LangGraph.",
            )

        assert context.latest_user_query == "А які його мінуси?"
        assert context.contextualized_user_query() == ("Поясни LangGraph.\n\nА які його мінуси?")
        captured_follow_up_contexts.append(context)
        return RouteDecision(
            route="research",
            confidence=1.0,
            reason="Technical follow-up resolved from recent context.",
            response_language="uk",
            resolved_query="Які мінуси LangGraph?",
        )

    def generate(
        context: ConversationContext,
        language: str,
        kind: ResponseKind,
    ) -> str:
        assert context.latest_user_query == "Поясни LangGraph."
        return "LangGraph — це фреймворк оркестрації."

    def research(
        query: str,
        language: str,
    ) -> ResearchResult:
        captured_research_queries.append(query)
        return ResearchResult(answer="Перевірені обмеження LangGraph.")

    with open_sqlite_checkpointer(tmp_path / "follow-up.sqlite3") as checkpointer:
        graph = build_core_graph(
            classify=classify,
            generate=generate,
            research=research,
            code=_unexpected_code,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": "follow-up-thread"}}

        graph.invoke(
            {"messages": [{"role": "user", "content": "Поясни LangGraph."}]},
            config=config,
        )
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "А які його мінуси?"}]},
            config=config,
        )

    assert result["route"] == "research"
    assert result["messages"][-1].content == "Перевірені обмеження LangGraph."
    assert captured_research_queries == ["Які мінуси LangGraph?"]
    assert len(captured_follow_up_contexts) == 1
    assert [message.role for message in captured_follow_up_contexts[0].messages] == [
        "user",
        "assistant",
        "user",
    ]


def test_managed_checkpointer_closes_connection_on_exit(tmp_path: Path) -> None:
    with open_sqlite_checkpointer(tmp_path / "managed.sqlite3") as checkpointer:
        checkpointer.conn.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        checkpointer.conn.execute("SELECT 1")


def test_checkpointer_isolates_conversation_threads(tmp_path: Path) -> None:
    checkpointer = create_sqlite_checkpointer(tmp_path / "isolated.sqlite3")
    graph = build_core_graph(
        classify=_classify_direct_answer,
        generate=_generate_response,
        research=_unexpected_research,
        code=_unexpected_code,
        checkpointer=checkpointer,
    )
    first_config = {"configurable": {"thread_id": "first-thread"}}
    second_config = {"configurable": {"thread_id": "second-thread"}}

    graph.invoke(
        {"messages": [{"role": "user", "content": "First thread question"}]},
        config=first_config,
    )
    graph.invoke(
        {"messages": [{"role": "user", "content": "Second thread question"}]},
        config=second_config,
    )

    first_messages = graph.get_state(first_config).values["messages"]
    second_messages = graph.get_state(second_config).values["messages"]

    assert [message.content for message in first_messages] == [
        "First thread question",
        "direct_answer:en:First thread question",
    ]
    assert [message.content for message in second_messages] == [
        "Second thread question",
        "direct_answer:en:Second thread question",
    ]

    checkpointer.conn.close()


def test_sqlite_checkpointer_serializes_research_sources(tmp_path: Path) -> None:
    source = SourceItem(
        title="LangGraph overview",
        url="https://docs.langchain.com/oss/python/langgraph/overview",
        source_type="documentation",
    )
    source_reference = SourceReference.from_source(source)
    checkpointer = create_sqlite_checkpointer(tmp_path / "research.sqlite3")
    graph = build_core_graph(
        classify=_classify_research,
        generate=_generate_response,
        research=lambda query, language: ResearchResult(
            answer=f"Research answer for {language}:{query}",
            sources=[source_reference],
        ),
        code=_unexpected_code,
        checkpointer=checkpointer,
    )

    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Research LangGraph"}]},
        config={"configurable": {"thread_id": "research-thread"}},
    )

    expected_result = ResearchResult(
        answer="Research answer for en:Research LangGraph",
        sources=[source_reference],
    ).to_record()
    assert result["agent_result"] == expected_result
    assert (
        graph.get_state({"configurable": {"thread_id": "research-thread"}}).values["agent_result"]
        == expected_result
    )

    checkpointer.conn.close()


def test_clarification_interrupt_resumes_with_persisted_state(tmp_path: Path) -> None:
    database_path = tmp_path / "checkpoints.sqlite3"
    first_checkpointer = create_sqlite_checkpointer(database_path)
    first_graph = build_core_graph(
        classify=_classify_clarification_then_direct_answer,
        generate=_generate_response,
        research=_unexpected_research,
        code=_unexpected_code,
        checkpointer=first_checkpointer,
    )
    config = {"configurable": {"thread_id": "clarification-thread"}}
    stale_result = AgentResult(
        answer="Stale answer",
        sources=[
            SourceReference.from_source(
                SourceItem(
                    title="Stale source",
                    url="https://example.com/stale",
                    source_type="web",
                )
            )
        ],
    )

    interrupted = first_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Help me with an agent.",
                }
            ],
            "agent_result": stale_result.to_record(),
        },
        config=config,
    )

    assert len(interrupted["__interrupt__"]) == 1
    assert interrupted["__interrupt__"][0].value == {
        "type": "clarification",
        "question": "What kind of agent are you building?",
    }
    assert interrupted["agent_result"] is None
    assert first_graph.get_state(config).values["agent_result"] is None

    first_checkpointer.conn.close()

    second_checkpointer = create_sqlite_checkpointer(database_path)
    second_graph = build_core_graph(
        classify=_classify_clarification_then_direct_answer,
        generate=_generate_response,
        research=_unexpected_research,
        code=_unexpected_code,
        checkpointer=second_checkpointer,
    )

    result = second_graph.invoke(
        Command(resume="A LangGraph agent with web search."),
        config=config,
    )

    assert result["route"] == "direct_answer"
    assert result["clarification_question"] is None
    assert result["messages"][-1].content == ("direct_answer:en:A LangGraph agent with web search.")

    second_checkpointer.conn.close()


def test_clarification_reprompts_for_empty_resume(tmp_path: Path) -> None:
    checkpointer = create_sqlite_checkpointer(tmp_path / "retry.sqlite3")
    graph = build_core_graph(
        classify=_classify_clarification_then_direct_answer,
        generate=_generate_response,
        research=_unexpected_research,
        code=_unexpected_code,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "clarification-retry-thread"}}

    graph.invoke(
        {"messages": [{"role": "user", "content": "Help me with an agent."}]},
        config=config,
    )
    retry = graph.invoke(Command(resume="   "), config=config)

    assert retry["__interrupt__"][0].value == {
        "type": "clarification",
        "question": "What kind of agent are you building?",
        "error": "Please provide a non-empty text answer.",
    }

    result = graph.invoke(
        Command(resume="A LangGraph agent with web search."),
        config=config,
    )

    assert result["route"] == "direct_answer"
    assert result["clarification_question"] is None

    checkpointer.conn.close()


def _classify_clarification_then_direct_answer(
    context: ConversationContext,
) -> RouteDecision:
    query = context.latest_user_query

    if query == "Help me with an agent.":
        return RouteDecision(
            route="clarification",
            confidence=0.9,
            reason="The request needs more detail.",
            response_language="en",
            resolved_query="Help with an agent.",
            clarification_question="What kind of agent are you building?",
        )

    assert query == "A LangGraph agent with web search."
    assert context.contextualized_user_query() == (
        "Help me with an agent.\n\nA LangGraph agent with web search."
    )

    return RouteDecision(
        route="direct_answer",
        confidence=0.95,
        reason="The clarified request is specific enough.",
        response_language="en",
        resolved_query="Help with a LangGraph agent that uses web search.",
    )


def _classify_direct_answer(context: ConversationContext) -> RouteDecision:
    query = context.latest_user_query
    return RouteDecision(
        route="direct_answer",
        confidence=1.0,
        reason=f"Direct response for {query}.",
        response_language="en",
        resolved_query=query,
    )


def _classify_research(context: ConversationContext) -> RouteDecision:
    query = context.latest_user_query
    return RouteDecision(
        route="research",
        confidence=1.0,
        reason=f"Research response for {query}.",
        response_language="en",
        resolved_query=query,
    )


def _generate_response(
    context: ConversationContext,
    language: str,
    kind: ResponseKind,
) -> str:
    return f"{kind}:{language}:{context.latest_user_query}"


def _unexpected_research(
    query: str,
    language: str,
) -> ResearchResult:
    raise AssertionError(f"Unexpected Research Agent call: {language}:{query}")


def _unexpected_code(
    query: str,
    language: str,
) -> CodeResult:
    raise AssertionError(f"Unexpected Code Agent call: {language}:{query}")
