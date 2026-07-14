# AI Research Assistant

A local multi-agent assistant for researching technologies, technical documentation,
Python libraries, APIs, GitHub projects, and software architecture decisions.

## Project status

The project is under active development.

Current milestone: repository and Python environment setup.

## Planned capabilities

- Request routing with LangGraph
- Research, Code, and Comparison agents
- Local LLM inference through Ollama
- Web, documentation, and GitHub search
- Source attribution
- Conversation checkpointing
- Human-in-the-Loop clarification
- Streamlit chat interface
- Optional LangSmith tracing

## Technology stack

- Python 3.11+
- LangGraph
- LangChain
- Ollama
- Streamlit
- Pydantic
- pytest
- Ruff
- mypy

## Local requirements

- Python 3.11 or newer
- Git
- GitHub Desktop
- Ollama

## Development roadmap

- [x] Create and publish the repository
- [x] Verify Python, Git, and Ollama
- [ ] Configure and test the Ollama integration
- [ ] Implement the core LangGraph router
- [ ] Add specialized agents
- [ ] Add persistence and Human-in-the-Loop
- [ ] Build the Streamlit interface
- [ ] Add observability and CI

## Security

Secrets must be stored in a local `.env` file and must never be committed.

Local databases, logs, virtual environments, caches, and generated runtime data are
excluded from version control.
