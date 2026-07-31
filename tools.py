"""
Test-level tools exposed to Gemini Live for sanity-checking function calling
over the ESP32 <-> server <-> Gemini bridge.
"""

import datetime

from google.genai import types


def get_current_time() -> dict:
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return {"current_time": now.strftime("%Y-%m-%d %H:%M:%S")}


def add_numbers(a: float, b: float) -> dict:
    """Adds two numbers together and returns the result."""
    return {"result": a + b}


TOOL_IMPLEMENTATIONS = {
    "get_current_time": get_current_time,
    "add_numbers": add_numbers,
}

TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_current_time",
                description="Get the current date and time.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                ),
            ),
            types.FunctionDeclaration(
                name="add_numbers",
                description="Add two numbers together and return the sum.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "a": types.Schema(
                            type=types.Type.NUMBER, description="The first number."
                        ),
                        "b": types.Schema(
                            type=types.Type.NUMBER, description="The second number."
                        ),
                    },
                    required=["a", "b"],
                ),
            ),
        ]
    )
]
