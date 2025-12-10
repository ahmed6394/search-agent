import os
import requests
from dotenv import load_dotenv
import ast
import operator as op

load_dotenv()

token = os.getenv("BRAVE_API")
url = "https://api.search.brave.com/res/v1/web/search"
base_params = {"count": 3}
headers = {"Accept": "application/json", "X-Subscription-Token": token}


def web_search(query: str) -> str:
    """
    Performs a web search using the Brave Search API and returns the response text.

    Args:
        query (str): The search query string

    Returns:
        str: The response text from the Brave Search API.
    """
    try:
        params = {"q": query, **base_params}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        return f"Search error: {str(e)}"
    except Exception as e:
        return f"Unexpected error during search: {str(e)}"


# Allowed operators for safe evaluation
_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_eval(node):
    """Recursively evaluate an AST node with a limited set of operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _ALLOWED_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _safe_eval(node.operand)
        return _ALLOWED_OPERATORS[type(node.op)](operand)
    raise ValueError("Unsupported expression")


def calculator(math_expression: str) -> str:
    """
    Safely evaluates a mathematical expression and returns the result as a string.

    This function safely evaluates basic math expressions without using eval().
    
    Args:
        math_expression (str): The mathematical expression to evaluate.
        
    Returns:
        str: The result of the evaluated expression.
    """
    try:
        node = ast.parse(math_expression, mode="eval").body
        result = _safe_eval(node)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


if __name__ == "__main__":
    query = "latest news on AI technology"
    print("Web Search Result:")
    print(web_search(query))
    print("\nCalculator Result:")
    print(calculator("2 + 2 * 2"))