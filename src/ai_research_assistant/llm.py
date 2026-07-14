from langchain_ollama import ChatOllama

from ai_research_assistant.config import get_settings


def create_chat_model() -> ChatOllama:
    """Create the configured Ollama chat model."""

    settings = get_settings()

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        validate_model_on_init=True,
    )
