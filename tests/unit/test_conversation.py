import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_research_assistant.conversation import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_MESSAGE_CHARS,
    MAX_CONTEXT_MESSAGES,
    TRUNCATION_MARKER,
    build_conversation_context,
)


def test_conversation_context_preserves_roles_and_user_topic() -> None:
    context = build_conversation_context(
        [
            SystemMessage(content="Internal system message"),
            HumanMessage(content="Explain LangGraph."),
            AIMessage(content="LangGraph is an orchestration framework."),
            HumanMessage(content="What are its limitations?"),
        ]
    )

    assert [message.role for message in context.messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert context.latest_user_query == "What are its limitations?"
    assert context.has_history is True
    assert context.contextualized_user_query() == (
        "Explain LangGraph.\n\nWhat are its limitations?"
    )
    assert "orchestration framework" not in context.contextualized_user_query()
    assert "orchestration framework" in context.render_reference_context()


def test_conversation_context_keeps_newest_messages_within_budget() -> None:
    messages = [
        HumanMessage(content=f"user-{index}-" + "x" * 990)
        if index % 2 == 0
        else AIMessage(content=f"assistant-{index}-" + "x" * 985)
        for index in range(7)
    ]

    context = build_conversation_context(messages)

    assert len(context.messages) <= MAX_CONTEXT_MESSAGES
    assert sum(len(message.content) for message in context.messages) <= MAX_CONTEXT_CHARS
    assert context.latest_user_query.startswith("user-6-")
    assert all("user-0-" not in message.content for message in context.messages)


def test_conversation_context_truncates_large_current_request() -> None:
    context = build_conversation_context([HumanMessage(content="start-" + "x" * 3_000 + "-end")])

    content = context.latest_user_query

    assert len(content) == MAX_CONTEXT_MESSAGE_CHARS
    assert content.startswith("start-")
    assert content.endswith("-end")
    assert TRUNCATION_MARKER in content


def test_conversation_context_requires_latest_user_message() -> None:
    with pytest.raises(ValueError, match="latest user message"):
        build_conversation_context([AIMessage(content="Assistant-only state")])
