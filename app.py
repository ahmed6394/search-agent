import asyncio
import chainlit as cl
from llm.agent import agent_query


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages from Chainlit UI"""
    
    try:
        # Run the synchronous agent_query in a thread to avoid blocking
        final_answer = await asyncio.to_thread(agent_query, message.content)
        
        # Send the response as a new message
        await cl.Message(content=final_answer).send()
        
    except RuntimeError as exc:
        # Fallback for event loop issues
        if "no running event loop" in str(exc).lower():
            final_answer = agent_query(message.content)
        else:
            final_answer = f"❌ Error: {exc}"
        await cl.Message(content=final_answer).send()
        
    except Exception as exc:
        final_answer = f"❌ Error: {exc}"
        await cl.Message(content=final_answer).send()