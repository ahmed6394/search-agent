"""
Main agent interface - entry point for querying.
Uses LangGraph for orchestration.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from .graph import build_agent_graph
from .state import AgentState

load_dotenv()


def agent_query(user_input: str) -> str:
    """
    Main entry point for querying the agent.
    
    Args:
        user_input: User's question or prompt
        
    Returns:
        str: Final answer from the agent
    """
    try:
        print(f"\n{'='*60}")
        print(f"User Input: {user_input}")
        print(f"{'='*60}")
        
        # Build and run graph
        graph = build_agent_graph()
        
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=user_input)],
            tool_calls_count=0,
            max_tool_calls=5,
        )
        
        # Execute graph (this runs the full loop)
        final_state = graph.invoke(initial_state)
        
        # Extract final answer from messages
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"Agent Response: {msg.content}")
                print(f"{'='*60}\n")
                return msg.content
        
        return "No response generated."
    
    except Exception as exc:
        error_msg = f"Error during agent query: {str(exc)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg


if __name__ == "__main__":
    # Test
    result = agent_query("What is 25 * 4?")
    print("Result:", result)