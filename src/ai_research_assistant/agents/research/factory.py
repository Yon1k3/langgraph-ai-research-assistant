from typing import cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from ai_research_assistant.agents.research.agent import ResearchAgent
from ai_research_assistant.agents.research.prompts import RESEARCH_SYSTEM_PROMPT
from ai_research_assistant.agents.research.synthesis import create_grounded_synthesizer
from ai_research_assistant.agents.research.tools import create_web_search_tool
from ai_research_assistant.agents.research.types import AgentRunner, SearchService
from ai_research_assistant.llm import create_chat_model
from ai_research_assistant.tools import create_lazy_tavily_search_service


def create_research_agent(
    search_service: SearchService | None = None,
    model: BaseChatModel | None = None,
) -> ResearchAgent:
    """Create the Ollama-backed Research Agent."""

    resolved_search_service = search_service or create_lazy_tavily_search_service()
    resolved_model = model or create_chat_model()
    search_tool = create_web_search_tool(resolved_search_service)
    synthesize = create_grounded_synthesizer(resolved_model)

    runner = cast(
        AgentRunner,
        create_agent(
            model=resolved_model,
            tools=[search_tool],
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            name="research_agent",
        ),
    )

    return ResearchAgent(
        runner=runner,
        search_tool=search_tool,
        synthesize=synthesize,
    )
