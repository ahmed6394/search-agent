from .agent import SearchAgent
import chainlit as cl
import os
from dotenv import load_dotenv

load_dotenv()
fastmcp_url = os.getenv("FASTMCP_URL")

cl.instrument_openai()

@cl.on_chat_start
async def on_chat_start():
    agent = SearchAgent()
    try:
        await agent.connect_fastmcp(fastmcp_url)
    except Exception as e:
        await cl.Message(content=f"Failed to connect to FastMCP at {fastmcp_url}: {e}").send()
        return
    cl.user_session.set("agent", agent)
    await cl.Message(content=f"Connected to FastMCP: {fastmcp_url} – {len(agent.tools)} tools found.").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    if agent is None:
        await cl.Message(content="Agent not initialized – please restart the chat.").send()
        return
    try:
        if message.elements:
            final_answer = await agent.run_chat_with_files(message.content, message.elements)
        else:
            final_answer = await agent.run_chat(message.content)
    except Exception as e:
        await cl.Message(content=f"Error processing request: {e}").send()
        return
    await cl.Message(content=final_answer).send()

@cl.on_chat_end
async def on_chat_end():
    agent = cl.user_session.get("agent")
    if agent:
        try:
            await agent.disconnect_fastmcp()
        except Exception:
            pass









