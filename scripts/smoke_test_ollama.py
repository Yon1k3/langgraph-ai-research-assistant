import json

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from ai_research_assistant.config import get_settings


class TechnologyInfo(BaseModel):
    """Structured information about a technology."""

    name: str = Field(description="Technology name")
    category: str = Field(description="Technology category")
    description: str = Field(description="One-sentence description")


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["city"],
        },
    },
}


def build_model() -> ChatOllama:
    """Create the configured Ollama chat model."""

    settings = get_settings()

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        validate_model_on_init=True,
    )


def run_chat_test(model: ChatOllama) -> None:
    """Check a normal chat response."""

    response = model.invoke(
        [
            ("system", "Follow the user's instruction exactly."),
            ("human", "Reply with exactly: OLLAMA_CHAT_OK"),
        ]
    )
    content = str(response.content).strip()

    if "OLLAMA_CHAT_OK" not in content:
        raise RuntimeError(f"Unexpected chat response: {content}")

    print("[PASS] Chat response")
    print(content)


def run_structured_output_test(model: ChatOllama) -> None:
    """Check JSON-schema structured output."""

    structured_model = model.with_structured_output(
        TechnologyInfo,
        method="json_schema",
    )
    result = structured_model.invoke(
        "Return structured information about LangGraph. "
        "Use LangGraph as the name, AI workflow framework as the category, "
        "and one short sentence as the description."
    )

    if not isinstance(result, TechnologyInfo):
        raise RuntimeError(f"Unexpected structured output: {result}")

    if result.name.lower() != "langgraph":
        raise RuntimeError(f"Unexpected technology name: {result.name}")

    print("\n[PASS] Structured output")
    print(result.model_dump_json(indent=2))


def run_tool_calling_test(model: ChatOllama) -> None:
    """Check whether the model creates a tool call."""

    tool_model = model.bind_tools([WEATHER_TOOL])
    response = tool_model.invoke(
        "Use the get_current_weather tool to look up the weather in Kyiv. "
        "You must call the tool and must not answer from memory."
    )

    if not response.tool_calls:
        raise RuntimeError(f"No tool call returned: {response.content}")

    tool_call = response.tool_calls[0]

    if tool_call["name"] != "get_current_weather":
        raise RuntimeError(f"Unexpected tool call: {tool_call}")

    city = str(tool_call["args"].get("city", "")).strip()

    if not city:
        raise RuntimeError(f"Tool call has no city: {tool_call}")

    print("\n[PASS] Tool calling")
    print(json.dumps(response.tool_calls, ensure_ascii=False, indent=2))


def main() -> None:
    """Run all Ollama capability checks."""

    print("Connecting to Ollama...")
    model = build_model()

    run_chat_test(model)
    run_structured_output_test(model)
    run_tool_calling_test(model)

    print("\nAll Ollama smoke tests passed.")


if __name__ == "__main__":
    main()
