import os
import sys
import json
from pathlib import Path
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
        
        # Load system prompt
        system_prompt_path = Path(__file__).resolve().parent.parent / "prompt" / "system_prompt.txt"
        self.system_msg = system_prompt_path.read_text(encoding="utf-8")
        
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_msg}
        ]
        
        # Define tools for the LLM
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
                    "description": "Evaluate a math expression with a safe parser.",
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

    def send_message(self, message: str, max_tool_calls: int = 5) -> str:
        """Send a message and handle tool calls if needed"""
        self.messages.append({"role": "user", "content": message})
        
        tool_call_count = 0
        
        while tool_call_count < max_tool_calls:
            print(f"[DEBUG] Making API call (attempt {tool_call_count + 1})")
            
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
            )

            choice = response.choices[0].message
            print(f"[DEBUG] Response stop_reason: {response.choices[0].finish_reason}")

            # Handle tool calls
            if choice.tool_calls:
                print(f"[DEBUG] Tool calls detected: {len(choice.tool_calls)}")
                
                # Add assistant response with tool calls
                self.messages.append({
                    "role": "assistant",
                    "content": choice.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in choice.tool_calls
                    ]
                })

                # Execute each tool call
                for tool_call in choice.tool_calls:
                    name = tool_call.function.name
                    args = tool_call.function.arguments or "{}"
                    
                    print(f"[DEBUG] Executing tool: {name}")
                    
                    try:
                        parsed = json.loads(args)
                        
                        if name == "web_search":
                            result = web_search(parsed.get("q", ""))
                            print(f"[DEBUG] Web search result length: {len(result)}")
                        elif name == "calculator":
                            result = calculator(parsed.get("math_expression", ""))
                            print(f"[DEBUG] Calculator result: {result}")
                        else:
                            result = f"Unknown tool: {name}"
                            
                    except Exception as exc:
                        result = f"Tool execution error: {exc}"
                        print(f"[DEBUG] Tool error: {exc}")

                    # Add tool result to conversation
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": str(result)[:2000],  # Limit result size
                    })

                tool_call_count += 1
                # Continue the loop to get the next response
                continue

            # No tool calls detected
            if choice.content:
                print(f"[DEBUG] Got final response from LLM")
                self.messages.append({"role": "assistant", "content": choice.content})
                return choice.content
            else:
                print(f"[DEBUG] Empty response, stopping")
                return "No response generated."

        return "Max tool calls reached. Unable to complete the request."


def agent_query(user_input: str) -> str:
    """
    Main entry point for querying the agent.
    
    Args:
        user_input: The user's question or prompt
        
    Returns:
        The agent's final answer
    """
    agent = Agent()
    
    try:
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        # Send the user input and get the response
        response = agent.send_message(user_input)
        print(f"Agent: {response}")
        print(f"{'='*60}\n")
        return response
        
    except Exception as exc:
        error_msg = f"Error during agent query: {str(exc)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg


if __name__ == "__main__":
    # Test the agent
    prompt = "What is the sum of the current temperature in Berlin and Paris?"
    result = agent_query(prompt)
    print("Final Result:", result)