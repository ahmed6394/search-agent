import os
from dotenv import load_dotenv
import logging
import httpx
import json
from .helper import *
from openai import AsyncOpenAI
from fastmcp.client import Client
from collections import OrderedDict


logger = logging.getLogger(__name__)
load_dotenv()
base_url = os.getenv("OPENAI_API_BASE_URL")
fastmcp_url = os.getenv("FASTMCP_URL")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not set in environment")

httpx_client = httpx.AsyncClient(http2=True, verify=False,
                                 timeout=httpx.Timeout(connect=10, read=120, write=30, pool=5))
AGENT_MAX_ITERATIONS = 5

class SearchAgent:
    def __init__(self):
        self.llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=httpx_client)
        self.fastmcp_client = None
        self.history = []
        self.tools = []
        # self.sandbox = AzureSessionManager()
        # --- token usage tracking ---
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.usage_events = OrderedDict()
        # --------------------------------
        system_prompt = load_system_prompt("sys_prompts/cookbook.md") or "You are a helpful research assistant."
        self.history.append({"role": "system", "content": system_prompt})
        logger.info("System prompt loaded and added to history.")

    async def connect_fastmcp(self, fastmcp_url: str):
        try:
            client = Client(fastmcp_url)
            self.fastmcp_client = await client.__aenter__()
            self.tools = await self.fastmcp_client.list_tools()
            logger.info(f"Connected to FastMCP: {fastmcp_url} – {len(self.tools)} tools found.")
            return self.tools
        except Exception as e:
            logger.exception("Failed to initialize FastMCP client")
            raise RuntimeError(f"Failed to connect to FastMCP server at {fastmcp_url}: {e}")

    async def activate_tool(self, tool_call, iteration: int):
        tool_name = tool_call.function.name
        args_raw = tool_call.function.arguments
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError as e:
            logger.warning(f"[Iteration {iteration}] Invalid JSON in arguments for tool {tool_name}: {args_raw}")
            args = {}
        logger.info(f"[Iteration {iteration}] calling tool {tool_name} with args={args}")
        
        try:
            tool_output = await self.fastmcp_client.call_tool(tool_name, args)
            content = tool_output.content[0].text if tool_output.content else ""
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = content
        except Exception as e:
            logger.exception(f"Tool {tool_name} failed: {e}")
            parsed = {"error": str(e)}
        return {
            "tool_name": tool_name,
            "parsed": parsed,
            "tool_call_id": tool_call.id
        }

    def tool_schema(self):
        return [
            {"type": "function",
             "function": {"name": ct.name,
                          "description": ct.description,
                          "parameters": ct.inputSchema,
                          },
            } for ct in self.tools
        ]

    def token_log(self, response, label: str):
        token_log(self, response, label)

    async def summarize_extracted_text(self, parsed: dict):
        text = parsed.get("text")
        if not text or not isinstance(text, str):
            logger.warning("No valid text to summarize.")
            parsed["summary"] = "(No extractable text for summary.)"
            return parsed
        
        summary_prompt = (
            "You are a concise summarization assistant. "
            "Summarize the provided text in 2–4 sentences. "
            "Focus on main facts, findings, or claims — omit menus, ads, navigation, unrelated links, or metadata. "
            "Do not speculate or add new information."
        )
        try:
            messages = [
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": text}
            ]
            response = await self.llm_client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
            )
            self.token_log(response, "summarization")
            summary = response.choices[0].message.content.strip()
            parsed["summary"] = summary or "(No summary produced.)"
            return parsed
        except Exception:
            logger.exception(f"Summarization failed. Input length: {len(text)}")
            parsed["summary"] = "(Summary failed due to an internal error.)"
            return parsed

    async def run_chat(self, user_input: str, max_iterations: int = AGENT_MAX_ITERATIONS):
        self.history.append({"role": "user", "content": user_input})
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            try:
                response = await self.llm_client.chat.completions.create(
                    model="gpt-5",
                    messages=self.history,
                    reasoning_effort="medium",
                    verbosity="low",
                    tools=self.tool_schema(),
                    tool_choice="auto",

                )
            except Exception as e:
                logger.exception("LLM request failed")
                return f"LLM request failed (Iteration {iteration}): {e}"
            self.token_log(response, label=f"iteration_{iteration}")
            assistant_msg = response.choices[0].message
            self.history.append(assistant_msg)
            if not assistant_msg.tool_calls:
                logger.info(f"Final answer returned at iteration {iteration}")
                return f"**Final Answer:**\n\n{assistant_msg.content}"

            all_citations = []
            for tool_call in assistant_msg.tool_calls:
                tool_result = await self.activate_tool(tool_call, iteration)
                tool_name = tool_result["tool_name"]
                parsed = tool_result.get("parsed")
                if tool_name == "web_fetch_extract":
                    parsed = await self.summarize_extracted_text(parsed)
                citations = extract_citations(parsed)
                all_citations.extend(citations)
                if tool_name == "web_fetch_extract" and isinstance(parsed, dict):
                    content_payload = {
                        "summary": parsed.get("summary", ""),
                        "url": parsed.get("url")
                    }
                else:
                    content_payload = parsed
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": json.dumps(content_payload),
                })
                await asyncio.sleep(0.5)
            self.history.append({
                "role": "user",
                "content": "Assess tool output. If evidence is adequate, answer now; otherwise, call another tool."
            })
        all_citations = list(dict.fromkeys(all_citations))
        self.history.append({
            "role": "user",
            "content": "Generate final answer from all tool results."
        })
        try:
            final_resp = await self.llm_client.chat.completions.create(
                model="gpt-5",
                messages=self.history,
            )
            self.token_log(final_resp, label="final_synthesis")
            final_answer = final_resp.choices[0].message.content
            return "**Final Answer (Max iterations reached):**\n\n" + final_answer
        except Exception as e:
            logger.exception("Final synthesis after max iterations failed")
            return f"Final synthesis failed: {e}"

    # async def run_chat_with_files(self, user_input: str, files: list, max_iterations: int = AGENT_MAX_ITERATIONS):
    #     file_texts = []
    #     for file in files:
    #         try:
    #             with open(file.path, "rb") as f:
    #                 content = f.read()
    #             text = extract_text_from_file(content, file.name)
    #             file_texts.append(f"Content from file '{file.name}':\n{text}\n")
    #         except Exception as e:
    #             logger.exception(f"Failed to read or extract text from file {file.name}")
    #             file_texts.append(f"Could not extract content from file '{file.name}': {e}\n")
    #     combined_file_text = "\n".join(file_texts)
    #     augmented_input = f"{user_input}\n\nHere are the contents of the uploaded files:\n{combined_file_text}"
    #     return await self.run_chat(augmented_input, max_iterations=max_iterations)

    async def run_chat_with_files(self, user_input: str, files: list, max_iterations: int = AGENT_MAX_ITERATIONS):
        sfile = []
        for file in files:
            local_path = file.path
            remote_path = f"./{file.name}"
            try:
                await self.sandbox.upload_file(local_path)
                sfile.append({"name": file.name, "path": remote_path})
            except Exception as e:
                sfile.append({"name": file.name, "path": None, "error": str(e)})
        sand_prompt = (
            f"{user_input}\n\n"
            "You are a GPT-5 based coding agent. "
            "The user has uploaded the following files into your Azure sandbox:\n"
            f"{json.dumps(sfile, indent=2)}\n\n"
            "You can write and run Python code INSIDE the sandbox to accomplish the user's request."
            "Open and process these files directly in your code."
        )
        return await self.run_chat(sand_prompt, max_iterations=max_iterations)

    async def disconnect_fastmcp(self):
        if not self.fastmcp_client:
            logger.debug("Error: FastMCP client was not connected.")
            return
        try:
            await self.fastmcp_client.__aexit__(None, None, None)
            logger.info("FastMCP client disconnected successfully.")
        except Exception:
            logger.debug("FastMCP client shutdown encountered an issue", exc_info=True)
        finally:
            self.fastmcp_client = None
            self.tools = []