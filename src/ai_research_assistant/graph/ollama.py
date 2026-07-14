from langchain_ollama import ChatOllama

from ai_research_assistant.graph.nodes import (
    ResponseGenerator,
    ResponseKind,
    RouteClassifier,
)
from ai_research_assistant.models import RouteDecision

ROUTER_SYSTEM_PROMPT = """
You are the routing component of a technical AI research assistant.

The supported domain is strictly limited to:

- software engineering and programming;
- artificial intelligence and machine learning;
- developer tools, frameworks, libraries, and APIs;
- technical architecture, documentation, and research.

Requests about nutrition, diets, meal planning, medicine, law, personal finance,
travel, entertainment, or general lifestyle are outside the supported domain.

Classify the latest user request into exactly one route using this priority:

1. unsupported:
   Use whenever the substantive request is outside the supported technical
   domain, even if it is simple or easy to answer.

2. clarification:
   Use when the request appears technical but is too ambiguous or incomplete
   to begin. Ask one concrete follow-up question.

3. comparison:
   Use for comparisons of two or more technologies, frameworks, libraries,
   tools, or architectural approaches.

4. code:
   Use for programming, APIs, integrations, debugging, code examples, or
   technical proof-of-concept requests.

5. research:
   Use for technical explanations, documentation, capabilities, limitations,
   versions, architecture, current information, or requests requiring sources.

6. direct_answer:
   Use only for greetings or casual messages without a substantive task, and
   for simple stable questions inside the supported technical domain.

Important boundaries:

- A request must not be routed to direct_answer merely because it is simple.
- Any substantive non-technical task must be routed to unsupported.
- A technical request that only says "help me" without describing the concrete
  problem, goal, technology, or failure must be routed to clarification.
- "Привіт!" is direct_answer.
- "Що таке Python?" is direct_answer.
- "Допоможи мені з агентом." is clarification. Ask what kind of agent the user
  is building and what specific help is needed.
- "Склади план харчування на тиждень." is unsupported.
- "Напиши FastAPI endpoint." is code.
- "Порівняй LangGraph і CrewAI." is comparison.
- "Поясни актуальні можливості LangGraph із джерелами." is research.

Output rules:

- Choose exactly one route.
- response_language must be a short ISO 639-1 language code matching the
  language of the user's request, for example "uk" or "en".
- confidence is a heuristic self-assessment from 0.0 to 1.0.
- reason must be one short technical sentence and must not expose hidden
  reasoning or chain of thought.
- clarification_question must contain one concrete question only when route is
  clarification.
- For every other route, clarification_question must be null.
""".strip()


DIRECT_ANSWER_SYSTEM_PROMPT = (
    "You are a concise technical assistant. Respond to the greeting, casual "
    "message, or simple stable technical question. Do not claim that you searched "
    "external sources. Respond entirely in the language represented by ISO code "
    "{language}."
)


STATIC_RESPONSES: dict[ResponseKind, dict[str, str]] = {
    "unsupported": {
        "uk": (
            "Цей асистент спеціалізується на програмуванні, AI та технічних "
            "дослідженнях, тому цей запит не підтримується."
        ),
        "en": (
            "This assistant specializes in programming, AI, and technical research, "
            "so this request is not supported."
        ),
    },
    "route_unavailable": {
        "uk": (
            "Запит розпізнано як технічний, але відповідний спеціалізований модуль "
            "ще не реалізований у поточній версії."
        ),
        "en": (
            "The request was recognized as technical, but the corresponding "
            "specialized workflow is not implemented in the current version yet."
        ),
    },
}


def create_ollama_route_classifier(model: ChatOllama) -> RouteClassifier:
    """Create a structured-output classifier backed by Ollama."""

    structured_model = model.with_structured_output(
        RouteDecision,
        method="json_schema",
    )

    def classify(query: str) -> RouteDecision:
        result = structured_model.invoke(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", query),
            ]
        )

        if not isinstance(result, RouteDecision):
            raise TypeError("Router returned an unexpected response type")

        return result

    return classify


def create_ollama_response_generator(model: ChatOllama) -> ResponseGenerator:
    """Create a response generator backed by Ollama."""

    def generate(
        query: str,
        language: str,
        kind: ResponseKind,
    ) -> str:
        if kind != "direct_answer":
            responses = STATIC_RESPONSES[kind]
            return responses.get(language, responses["en"])

        system_prompt = DIRECT_ANSWER_SYSTEM_PROMPT.format(language=language)

        response = model.invoke(
            [
                ("system", system_prompt),
                ("human", query),
            ]
        )

        if not isinstance(response.content, str):
            raise TypeError("Response node returned non-text content")

        content = response.content.strip()

        if not content:
            raise ValueError("Response node returned empty content")

        return content

    return generate
