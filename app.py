
import chainlit as cl
from test_api import res
cl.instrument_openai()

@cl.on_message
async def on_message(message: cl.Message):
    response1 = await res(message.content)
    if response1 is not None:
        await cl.Message(content=response1).send()