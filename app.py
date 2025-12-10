import asyncio
import chainlit as cl
from llm.agent import agent_query


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages from Chainlit UI"""
    
    try:
        # Run the agent_query (now with LangGraph) in a thread
        final_answer = await asyncio.to_thread(agent_query, message.content)
        
        # Send the response
        await cl.Message(content=final_answer).send()
        
    except Exception as exc:
        error_msg = f"❌ Error: {exc}"
        print(f"[ERROR] {error_msg}")
        await cl.Message(content=error_msg).send()