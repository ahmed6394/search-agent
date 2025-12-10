
import asyncio
import chainlit as cl
from llm.agent import agent_query


@cl.on_message
async def on_message(message: cl.Message):
    # Run the synchronous agent_query in a thread when possible; fall back to direct call if no loop.
    try:
        final_answer = await asyncio.to_thread(agent_query, message.content)
    except RuntimeError as exc:
        # If no running loop is available (e.g., invoked outside Chainlit), run synchronously.
        if "no running event loop" in str(exc).lower():
            final_answer = agent_query(message.content)
        else:
            final_answer = f"Error: {exc}"
    except Exception as exc:
        final_answer = f"Error: {exc}"

    await cl.Message(content=final_answer).send()