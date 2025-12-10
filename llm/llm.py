import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI


class Agent:
    def __init__(self):

        load_dotenv()
        api_key = os.getenv("FREE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("FREE_OPENAI_API_KEY is not set in the environment")

        # Use OpenRouter endpoint with the provided key.
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.system_msg = "You are a helpful assistant."
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_msg}
        ]

    def send_message(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=self.messages,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    agent = Agent()
    message = "Write a 20 word explaination of how a search agent works."
    response = agent.send_message(message)
    print("Response from agent:", response)