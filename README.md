# AI Agent with LangGraph

A production-ready agent system using LangGraph for orchestration.

## Architecture
┌─────────────────┐
│   User Input    │
└────────┬────────┘
│
▼
┌─────────────────┐
│  agent_node     │ (LLM decides: tool call or respond?)
└────────┬────────┘
│
▼
┌─────────────────┐
│ should_continue │ (Router: continue or end?)
└─┬──────────────┬┘
│              │
yes            no
│              │
▼              ▼
┌──────────┐  ┌─────┐
│tool_node │  │ END │
└────┬─────┘
│
└──────> back to agent_node

## Installation
```bash
uv sync
```

## Usage

### Command Line
```python
from llm.agent import agent_query

result = agent_query("What is 2 + 2?")
print(result)
```

### Web UI (Chainlit)
```bash
chainlit run app.py -w
```

## File Structure

- `llm/agent.py` - Main entry point
- `llm/graph.py` - LangGraph state machine definition
- `llm/nodes.py` - Individual node implementations
- `llm/state.py` - State schema (TypedDict)
- `tools/tools.py` - Tool implementations (web_search, calculator)
- `app.py` - Chainlit UI integration
- `prompt/system_prompt.txt` - System prompt

## Key Features

✅ LangGraph for robust orchestration
✅ Multi-tool support (web_search, calculator)
✅ State persistence across calls
✅ Configurable max tool calls
✅ Chainlit UI integration
✅ Comprehensive logging