from collections.abc import Sequence
from typing import Literal, Self

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CONTEXT_MESSAGES = 6
MAX_CONTEXT_CHARS = 4_000
MAX_CONTEXT_MESSAGE_CHARS = 1_500
TRUNCATION_MARKER = "\n...[truncated]...\n"


class ConversationMessage(BaseModel):
    """One bounded user or assistant message used as short-term context."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ConversationContext(BaseModel):
    """Bounded recent dialogue passed to routing and response generation."""

    messages: tuple[ConversationMessage, ...] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_latest_message(self) -> Self:
        """Require the current request to be the final context message."""

        if self.messages[-1].role != "user":
            raise ValueError("conversation context must end with a user message")

        return self

    @classmethod
    def from_query(cls, query: str) -> Self:
        """Create a one-message context for standalone agent usage."""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("conversation query must not be empty")

        return cls(
            messages=(
                ConversationMessage(
                    role="user",
                    content=_truncate_message(
                        normalized_query,
                        MAX_CONTEXT_MESSAGE_CHARS,
                    ),
                ),
            )
        )

    @property
    def latest_user_query(self) -> str:
        """Return the current user request."""

        return self.messages[-1].content

    @property
    def has_history(self) -> bool:
        """Return whether an earlier dialogue message is available."""

        return len(self.messages) > 1

    def contextualized_user_query(self) -> str:
        """Render user turns without treating prior assistant text as evidence."""

        return "\n\n".join(message.content for message in self.messages if message.role == "user")

    def render_reference_context(self) -> str:
        """Render dialogue for reference resolution, never as factual evidence."""

        rendered_messages = [
            f"{message.role.upper()}: {message.content}" for message in self.messages
        ]
        return "\n\n".join(rendered_messages)


def build_conversation_context(
    messages: Sequence[BaseMessage],
) -> ConversationContext:
    """Select the newest bounded text messages from LangGraph state."""

    candidates: list[ConversationMessage] = []

    for message in messages:
        if message.type not in {"human", "ai"}:
            continue

        if not isinstance(message.content, str):
            raise TypeError("Conversation context only supports text messages")

        normalized_content = message.content.strip()

        if not normalized_content:
            continue

        candidates.append(
            ConversationMessage(
                role="user" if message.type == "human" else "assistant",
                content=normalized_content,
            )
        )

    if not candidates or candidates[-1].role != "user":
        raise ValueError("Conversation context requires a latest user message")

    selected_reversed: list[ConversationMessage] = []
    remaining_chars = MAX_CONTEXT_CHARS

    for candidate in reversed(candidates[-MAX_CONTEXT_MESSAGES:]):
        if remaining_chars <= 0:
            break

        content_limit = min(MAX_CONTEXT_MESSAGE_CHARS, remaining_chars)
        bounded_content = _truncate_message(candidate.content, content_limit)

        if not bounded_content:
            continue

        selected_reversed.append(
            ConversationMessage(
                role=candidate.role,
                content=bounded_content,
            )
        )
        remaining_chars -= len(bounded_content)

    return ConversationContext(messages=tuple(reversed(selected_reversed)))


def _truncate_message(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content

    if max_chars <= len(TRUNCATION_MARKER):
        return content[:max_chars]

    available_chars = max_chars - len(TRUNCATION_MARKER)
    head_chars = available_chars // 2
    tail_chars = available_chars - head_chars

    return f"{content[:head_chars]}{TRUNCATION_MARKER}{content[-tail_chars:]}"
