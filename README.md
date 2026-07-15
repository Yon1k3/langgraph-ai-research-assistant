# AI Research Assistant

A local multi-agent assistant for researching technologies, technical documentation,
Python libraries, APIs, GitHub projects, and software architecture decisions.

## Project status

The project is under active development.

Current milestone: grounded Research Agent with claim-level source verification.

Implemented so far:

- local Ollama chat, structured output, and tool calling;
- LangGraph request router with direct, unsupported, and research paths;
- Tavily-backed Research Agent with stable evidence IDs, verified supporting quotes,
  and deterministic source attribution;
- SQLite short-term conversation persistence by `thread_id`;
- clarification pause and resume through LangGraph interrupts.

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
- [x] Configure and test the Ollama integration
- [x] Implement the core LangGraph router
- [ ] Add all specialized agents (Research Agent is complete)
- [x] Add persistence and Human-in-the-Loop clarification
- [ ] Build the Streamlit interface
- [ ] Add observability and CI

## Security

Secrets must be stored in a local `.env` file and must never be committed.

Local databases, logs, virtual environments, caches, and generated runtime data are
excluded from version control.

The local checkpoint database path is configured through `CHECKPOINT_DB_PATH` and
defaults to `data/checkpoints.sqlite3`.
