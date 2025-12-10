"""
LLM Nodes for Agent Pipeline
Each function is a "node" that processes state and returns updated state.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from tools.tools import web_search, calculator
from .state import AgentState

load_dotenv()


def get_llm():
    """Initialize LLM with tools"""
    api_key = os.getenv("FREE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("FREE_OPENAI_API_KEY not set")
    
    return ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
    )


def get_system_prompt():
    """Load system prompt from file"""
    prompt_path = Path(__file__).resolve().parent.parent / "prompt" / "system_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def agent_node(state: AgentState) -> dict:
    """
    NODE 1: LLM Agent Node
    - Takes messages from state
    - LLM decides: should I use tools or respond?
    - Returns updated state with new message
    """
    print(f"[AGENT] Processing {len(state['messages'])} messages")
    
    llm = get_llm()
    system_prompt = get_system_prompt()
    
    # Define tools for LLM binding
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using Brave API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate mathematical expressions safely",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "math_expression": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        }
                    },
                    "required": ["math_expression"],
                },
            },
        },
    ]
    
    llm_with_tools = llm.bind_tools(tools_schema)
    
    # Add system prompt to messages
    messages_with_system = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]
    
    response = llm_with_tools.invoke(messages_with_system)
    print(f"[AGENT] LLM response - tool_calls: {len(getattr(response, 'tool_calls', []))}")
    
    # Return updated state
    return {
        "messages": state["messages"] + [response],
        "tool_calls_count": state["tool_calls_count"],
        "max_tool_calls": state["max_tool_calls"],
    }


def tool_executor_node(state: AgentState) -> dict:
    """
    NODE 2: Tool Executor Node
    - Extracts tool calls from last message
    - Executes each tool
    - Returns ToolMessage results to state
    """
    last_message = state["messages"][-1]
    
    # Check if there are tool calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        print("[TOOLS] No tool calls found")
        return state
    
    print(f"[TOOLS] Executing {len(last_message.tool_calls)} tool calls")
    
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.name
        tool_input = tool_call.args
        tool_call_id = tool_call.id
        
        print(f"[TOOLS] Running: {tool_name}")
        
        try:
            if tool_name == "web_search":
                result = web_search(tool_input.get("q", ""))
            elif tool_name == "calculator":
                result = calculator(tool_input.get("math_expression", ""))
            else:
                result = f"Unknown tool: {tool_name}"
        
        except Exception as e:
            result = f"Tool execution error: {str(e)}"
            print(f"[TOOLS] Error: {e}")
        
        # Create ToolMessage for each result
        tool_message = ToolMessage(
            name=tool_name,
            content=str(result)[:2000],  # Limit response size
            tool_call_id=tool_call_id,
        )
        tool_results.append(tool_message)
    
    # Return state with tool results added
    return {
        "messages": state["messages"] + tool_results,
        "tool_calls_count": state["tool_calls_count"] + 1,
        "max_tool_calls": state["max_tool_calls"],
    }


def should_continue(state: AgentState) -> str:
    """
    ROUTER: Conditional logic
    - Check if last message has tool calls
    - Check if we haven't exceeded max calls
    - Route to "tools" or "end"
    """
    last_message = state["messages"][-1]
    tool_calls_exist = hasattr(last_message, "tool_calls") and last_message.tool_calls
    under_limit = state["tool_calls_count"] < state["max_tool_calls"]
    
    if tool_calls_exist and under_limit:
        print(f"[ROUTER] Continue to tools (call {state['tool_calls_count'] + 1}/{state['max_tool_calls']})")
        return "continue"
    else:
        reason = "limit reached" if not under_limit else "no tool calls"
        print(f"[ROUTER] End ({reason})")
        return "end"