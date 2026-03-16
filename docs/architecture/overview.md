# Architecture Overview

## System Layers

```
┌─────────────────────────────────────────┐
│  UI Layer (Chainlit)                    │  Entry point
├─────────────────────────────────────────┤
│  Agent Orchestration (teams.py)         │  Multi-agent coordination
├─────────────────────────────────────────┤
│  Specialized Agents (factories.py)      │  7 runtime agents + 1 evaluation agent
├─────────────────────────────────────────┤
│  Tools + Storage (toolkits + S3)        │  Domain logic & persistence
└─────────────────────────────────────────┘
```

## Entry Point

**Chainlit** (`chainlit_app.py`):

- WebSocket-based real-time chat interface
- Per-session agent teams with authentication
- Tool call visualization as steps
- SMILES to inline molecule images
- Streaming response display
- Chainlit persistence disabled by default
- File upload support with S3 integration

## Streaming Response Pattern

Chainlit uses streaming for real-time display:

```python
for chunk in agent.run(prompt, stream=True):
    if is_tool_event(chunk):
        display_as_step(chunk)  # Tool calls shown as Chainlit Steps
    elif is_text_chunk(chunk):
        stream_to_ui(chunk)     # Text streamed to message
```

This allows users to see progress as agents work, rather than waiting for completion.

## Agent State Management

Agents use `session_state` (a persistent dict) to pass data between runs and between agents:

```python
# Save in one agent
agent.session_state["data_path"] = "results.csv"

# Access in another agent (same team)
path = agent.session_state.get("data_path")
```

## Agent Coordination

The `get_cs_copilot_agent_team()` function in `teams.py` creates a coordinated team:

```python
team = get_cs_copilot_agent_team(model)
# Creates Team with:
# - 7 runtime agents
# - Shared SqliteDb for memory persistence
# - Context management (num_history_runs=5)
# - Member interaction sharing
# - Streaming event propagation
```

Capabilities:

- **Multi-Agent Memory**: Session history persisted in SQLite
- **Context Sharing**: Agents access each other's outputs via `session_state`
- **Streaming**: Real-time event propagation from member agents to UI
- **Agentic State**: SQLite-backed memory and recent session history for coordinated workflows
