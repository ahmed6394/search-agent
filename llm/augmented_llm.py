import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI
from tools.tools import web_search, calculator


class Agent:
    def __init__(self):

        load_dotenv()
        api_key = os.getenv("FREE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("FREE_OPENAI_API_KEY is not set in the environment")

        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        system_prompt_path = Path(__file__).resolve().parent.parent / "prompt" / "system_prompt.txt"
        self.system_msg = system_prompt_path.read_text(encoding="utf-8")
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_msg}
        ]

    def send_message(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content