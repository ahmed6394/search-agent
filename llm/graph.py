"""
LangGraph state machine definition.
Defines the flow: agent -> router -> tools -> agent -> ... -> end
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import agent_node, tool_executor_node, should_continue


def build_agent_graph():
    """
    Construct the LangGraph state machine.
    
    Flow:
    START -> agent_node -> should_continue (router)
                              ├─ "continue" -> tool_executor_node -> agent_node (loop)
                              └─ "end" -> END
    """
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_executor_node)
    
    # Add edges
    workflow.add_edge(START, "agent")  # Always start at agent
    
    # Conditional routing from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",      # If tool calls, go to tools
            "end": END,               # Otherwise end
        }
    )
    
    # Tools always route back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


if __name__ == "__main__":
    # Test graph structure
    graph = build_agent_graph()
    print(graph.get_graph().draw_ascii())