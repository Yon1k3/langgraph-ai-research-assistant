import re

import httpx
from langchain_core.exceptions import OutputParserException
from ollama import ResponseError

from ai_research_assistant.models import ErrorCategory, ErrorInfo
from ai_research_assistant.tools.web_search import (
    InvalidSearchResponseError,
    SearchAuthenticationError,
    SearchConfigurationError,
    SearchRateLimitError,
    SearchUnavailableError,
)

UKRAINIAN_TEXT_PATTERN = re.compile(r"[іїєґІЇЄҐ]")

ERROR_MESSAGES: dict[ErrorCategory, dict[str, str]] = {
    "model_unavailable": {
        "uk": (
            "Не вдалося підключитися до локальної моделі Ollama. "
            "Перевір, чи Ollama запущена, і спробуй ще раз."
        ),
        "en": (
            "The local Ollama model is unavailable. Check that Ollama is running and try again."
        ),
    },
    "invalid_model_output": {
        "uk": ("Модель повернула некоректну відповідь. Спробуй переформулювати запит."),
        "en": ("The model returned an invalid response. Try rephrasing the request."),
    },
    "search_not_configured": {
        "uk": ("Вебпошук не налаштований. Додай WEB_SEARCH_API_KEY до локального файлу .env."),
        "en": ("Web search is not configured. Add WEB_SEARCH_API_KEY to the local .env file."),
    },
    "search_authentication": {
        "uk": "Сервіс вебпошуку відхилив API-ключ. Перевір локальний файл .env.",
        "en": "The web search service rejected the API key. Check the local .env file.",
    },
    "search_rate_limit": {
        "uk": "Ліміт вебпошуку вичерпано. Спробуй ще раз пізніше.",
        "en": "The web search limit has been reached. Try again later.",
    },
    "search_unavailable": {
        "uk": "Вебпошук тимчасово недоступний. Спробуй ще раз пізніше.",
        "en": "Web search is temporarily unavailable. Try again later.",
    },
    "invalid_search_response": {
        "uk": "Вебпошук повернув некоректні дані. Спробуй ще раз пізніше.",
        "en": "Web search returned invalid data. Try again later.",
    },
    "no_trustworthy_sources": {
        "uk": (
            "Не вдалося знайти достатньо надійних джерел. "
            "Уточни технологію, версію або потрібну можливість."
        ),
        "en": (
            "Not enough trustworthy sources were found. "
            "Specify the technology, version, or capability you need."
        ),
    },
}


class ModelUnavailableError(RuntimeError):
    """Raised when the configured local model cannot be reached or loaded."""


class InvalidModelOutputError(RuntimeError):
    """Raised when a model response does not satisfy the application contract."""


class InvalidResearchResultError(RuntimeError):
    """Raised when the research workflow returns invalid evidence or state."""


class ResearchSearchError(RuntimeError):
    """Raised when mandatory research search produces no usable sources."""


def detect_fallback_language(text: str) -> str:
    """Choose a safe supported language when routing fails before classification."""

    return "uk" if UKRAINIAN_TEXT_PATTERN.search(text) else "en"


def map_runtime_error(error: Exception, language: str) -> ErrorInfo | None:
    """Map known runtime failures to safe user-facing metadata."""

    if isinstance(error, SearchConfigurationError):
        category: ErrorCategory = "search_not_configured"
    elif isinstance(error, SearchAuthenticationError):
        category = "search_authentication"
    elif isinstance(error, SearchRateLimitError):
        category = "search_rate_limit"
    elif isinstance(error, SearchUnavailableError):
        category = "search_unavailable"
    elif isinstance(error, InvalidSearchResponseError):
        category = "invalid_search_response"
    elif isinstance(error, ResearchSearchError):
        category = "no_trustworthy_sources"
    elif isinstance(
        error,
        (
            InvalidModelOutputError,
            InvalidResearchResultError,
            OutputParserException,
        ),
    ):
        category = "invalid_model_output"
    elif isinstance(
        error,
        (
            ModelUnavailableError,
            ConnectionError,
            TimeoutError,
            httpx.HTTPError,
            ResponseError,
        ),
    ):
        category = "model_unavailable"
    else:
        return None

    resolved_language = language if language in {"uk", "en"} else "en"
    return ErrorInfo(
        category=category,
        message=ERROR_MESSAGES[category][resolved_language],
    )
