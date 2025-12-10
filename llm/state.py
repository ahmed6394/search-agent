from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    Shared state that persists across all agent operations.
    LangGraph manages this state automatically.
    """
    messages: List[BaseMessage]
    tool_calls_count: int
    max_tool_calls: int