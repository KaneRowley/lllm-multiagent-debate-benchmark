
import json
import os

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

CALCULATOR_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a precise arithmetic calculation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                        "operation": {
                            "type": "string",
                            "enum": [
                                "add",
                                "subtract",
                                "multiply",
                                "divide"
                            ]
                        }
                    },
                    "required": ["a", "b", "operation"],
                    "additionalProperties": False
                }
            }
        }
    ]

def calculate(a: float, b: float, operation: str):
    """Perform one exact arithmetic operation."""

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    raise ValueError(f"Unknown operation: {operation}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    # Get the model name.
    model = os.getenv("MODEL_ID", "qwen3:4b")

    # Connect to Ollama's OpenAI-compatible API.
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )

    messages = [
        {
            "role": "user",
            "content": (
                "A shop sells 17 boxes of pencils. "
                "Each box contains 24 pencils. "
                "You must use the calculate tool to multiply "
                "17 by 24, then give me the final answer."
            )
        }]

    # Ask the model to make a tool call.
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=CALCULATOR_TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        print("Model answered without using a tool.")
        print(assistant_message.content)
        return

    # Record the model's requested tool call.
    messages.append(
        assistant_message.model_dump(exclude_none=True)
    )

    # Execute the requested function.
    for call in assistant_message.tool_calls:
        if call.function.name != "calculate":
            raise ValueError("Unknown tool requested.")

        arguments = json.loads(call.function.arguments)

        result = calculate(**arguments)

        print("Tool requested:", call.function.name)
        print("Arguments:", arguments)
        print("Calculator result:", result)

        # Send the tool's result back to the model.
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": str(result),
        })

    # Let the model interpret the tool result.
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    print("\nFinal answer:")
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    main()
