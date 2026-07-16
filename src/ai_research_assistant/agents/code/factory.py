from typing import cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from ai_research_assistant.agents.code.agent import CodeAgent
from ai_research_assistant.agents.code.prompts import CODE_AGENT_SYSTEM_PROMPT
from ai_research_assistant.agents.code.synthesis import create_code_synthesizer
from ai_research_assistant.agents.code.types import AgentRunner
from ai_research_assistant.llm import create_chat_model
from ai_research_assistant.tools import (
    GitHubSearchService,
    OfficialDocumentationSearchService,
    create_github_search_service,
    create_github_tools,
    create_official_documentation_search_service,
    create_official_documentation_tool,
)


def create_code_agent(
    model: BaseChatModel | None = None,
    documentation_service: OfficialDocumentationSearchService | None = None,
    github_service: GitHubSearchService | None = None,
) -> CodeAgent:
    """Create the Ollama-backed Code Agent with read-only source tools."""

    resolved_model = model or create_chat_model()
    resolved_documentation_service = (
        documentation_service or create_official_documentation_search_service()
    )
    resolved_github_service = github_service or create_github_search_service()

    documentation_tool = create_official_documentation_tool(resolved_documentation_service)
    github_tools = create_github_tools(resolved_github_service)
    tools = [documentation_tool, *github_tools]
    repository_search_tool = next(
        tool for tool in github_tools if tool.name == "search_github_repositories"
    )
    readme_tool = next(tool for tool in github_tools if tool.name == "read_github_readme")
    code_search_tool = next(tool for tool in github_tools if tool.name == "search_github_code")

    runner = cast(
        AgentRunner,
        create_agent(
            model=resolved_model,
            tools=tools,
            system_prompt=CODE_AGENT_SYSTEM_PROMPT,
            name="code_agent",
        ),
    )

    return CodeAgent(
        runner=runner,
        initial_search_tool=repository_search_tool,
        synthesize=create_code_synthesizer(resolved_model),
        documentation_tool=documentation_tool,
        readme_tool=readme_tool,
        code_search_tool=code_search_tool,
    )
