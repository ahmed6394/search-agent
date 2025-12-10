import os
import sys
from pathlib import Path
import re
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Perform a web search via Brave API.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "Search query string",
                            }
                        },
                        "required": ["q"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression with a safe parser (no eval).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "math_expression": {
                                "type": "string",
                                "description": "Math expression to evaluate",
                            }
                        },
                        "required": ["math_expression"],
                    },
                },
            },
        ]

    def send_message(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
        )

        choice = response.choices[0].message

        if choice.tool_calls:
            for tool_call in choice.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments or "{}"
                fn = known_actions.get(name)
                if not fn:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        import json

                        parsed = json.loads(args)
                        if name == "web_search":
                            result = fn(parsed.get("q", ""))
                        elif name == "calculator":
                            result = fn(parsed.get("math_expression", ""))
                        else:
                            result = f"Unhandled tool: {name}"
                    except Exception as exc:
                        result = f"Tool execution error: {exc}"

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": str(result),
                    }
                )

            follow_up = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=self.messages,
                tool_choice="none",
            )
            final_msg = follow_up.choices[0].message.content
            self.messages.append({"role": "assistant", "content": final_msg})
            return final_msg

        self.messages.append({"role": "assistant", "content": choice.content})
        return choice.content

known_actions = {
    "web_search": web_search,
    "calculator": calculator,
}

def extract_action(message: str):
    """Parse a single-line Action and Action Input. Ignores suspicious/malformed lines."""
    action = None
    action_input = None
    for line in message.splitlines():
        line = line.strip()
        if line.startswith("Action Input:"):
            action_input = line.removeprefix("Action Input:").strip()
        elif line.startswith("Action:"):
            candidate = line.removeprefix("Action:").strip()
            if candidate in known_actions:
                action = candidate
    return action, action_input


def extract_answer(message: str):
    """Return the first line starting with 'Answer:'; otherwise None."""
    for line in message.splitlines():
        line = line.strip()
        if line.startswith("Answer:"):
            return line.removeprefix("Answer:").strip()
    return None

def agent_query(user_input, max_turns=10):
    agent = Agent()
    current_input = user_input

    last_response = None
    turns = 0
    idle_turns = 0
    while turns < max_turns:
        turns += 1
        response = agent.send_message(current_input)
        print("Agent Response:", response)
        print("-----")

        # Stop if we're not making progress.
        if response == last_response:
            return "No progress detected; stopping."
        last_response = response

        action, action_input = extract_action(response)

        if action:
            if action_input is None:
                return "No action input provided; stopping."
            result = known_actions[action](action_input)
            current_input = str(result)
            idle_turns = 0
            continue

        answer = extract_answer(response)
        if answer is not None:
            return answer

        idle_turns += 1
        if idle_turns >= 2:
            return "No answer after repeated prompts; stopping."

        # No action and no explicit answer; nudge the model for a final reply.
        current_input = "No answer yet. Please provide the final answer."

    return "Max turns reached without completion." 


if __name__ == "__main__":
    prompt = "What is the sum of the current temperature in Berlin and Paris?"
    print("User:", prompt)
    print("Assistant:", agent_query(prompt))