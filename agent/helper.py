
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

def load_system_prompt(filename: str):
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except Exception as e:
        logger.exception(f"Failed to load system prompt: {e}")
        return "You are helpful research assistant."

def extract_citations(parsed: Any):
    urls = set()
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "url" in item:
                urls.add(item["url"])
    elif isinstance(parsed, dict) and "url" in parsed:
        urls.add(parsed["url"])
    return list(urls)

def token_log(agent, response, label: str):
    input_tokens = output_tokens = total_tokens = 0
    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

    agent.total_input_tokens += input_tokens
    agent.total_output_tokens += output_tokens
    agent.total_tokens += total_tokens
    agent.usage_events[label] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens
    }
    logger.info(
        f"[Usage:{label}] prompt={input_tokens} completion={output_tokens} total={total_tokens} "
        f"(cumulative total={agent.total_tokens})"
    )

async def summarize_fetch_results(agent, fetched_results):
    tasks = [
        agent.summarize_extracted_text(res["parsed"])
        for res in fetched_results
        if res["tool_name"] == "web_fetch_extract"
    ]
    summaries = await asyncio.gather(*tasks, return_exceptions=True)
    return summaries

async def parallel_activate_tools(agent, tool_calls, iteration: int):
    tasks = [agent.activate_tool(tc, iteration) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    parsed_results = []
    for r in results:
        if isinstance(r, Exception):
            continue
        parsed_results.append(r)
    return parsed_results

def extract_tool_calls_by_name(tool_calls, name):
    return [
        tc for tc in tool_calls
        if getattr(getattr(tc, "function", None), "name", "") == name
    ]

def extract_text_from_file(file_content: bytes, file_name: str):
    try:
        if file_name.endswith(".txt") or file_name.endswith(".md"):
            return file_content.decode("utf-8", errors="ignore")

        elif file_name.endswith(".docx"):
            from docx import Document
            from io import BytesIO
            doc = Document(BytesIO(file_content))
            return "\n".join([p.text for p in doc.paragraphs])

        else:
            logger.warning(f"Unsupported file type for {file_name}. Returning empty content.")
            return ""

    except Exception as e:
        logger.exception(f"Failed to extract text from {file_name}: {e}")
        return ""