from email.mime import message
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("FREE_OPENAI_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

async def res(message: str):
    response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
            {
                "content": "You are a helpful bot, you always reply in English",
                "role": "system"
            },
            {
                "content": message,
                "role": "user"
            }
        ],
    )
    return response.choices[0].message.content